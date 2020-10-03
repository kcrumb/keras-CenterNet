"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import numpy as np
import progressbar
import argparse
import sys
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import os
from timeit import default_timer as timer

from utils.compute_overlap import compute_overlap
from utils.visualization import draw_detections, draw_annotations
from generators.utils import get_affine_transform, affine_transform

assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).

    Returns:
        The average precision as computed in py-faster-rcnn.

    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, visualize=False,
                    flip_test=False,
                    keep_resolution=False):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]

    total_time = 0
    img_count = 0

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        image = generator.load_image(i)
        src_image = image.copy()

        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        s = max(image.shape[0], image.shape[1]) * 1.0

        if not keep_resolution:
            tgt_w = generator.input_size
            tgt_h = generator.input_size
            image = generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
        else:
            tgt_w = image.shape[1] | 31 + 1
            tgt_h = image.shape[0] | 31 + 1
            image = generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
        if flip_test:
            flipped_image = image[:, ::-1]
            inputs = np.stack([image, flipped_image], axis=0)
        else:
            inputs = np.expand_dims(image, axis=0)
        # run network
        img_count += 1
        start = timer()
        detections = model.predict_on_batch(inputs)[0]
        end = timer()
        total_time += (end - start)
        scores = detections[:, 4]
        # select indices which have a score above the threshold
        indices = np.where(scores > score_threshold)[0]

        # select those detections
        detections = detections[indices]
        detections_copy = detections.copy()
        detections = detections.astype(np.float64)
        trans = get_affine_transform(c, s, (tgt_w // 4, tgt_h // 4), inv=1)

        for j in range(detections.shape[0]):
            detections[j, 0:2] = affine_transform(detections[j, 0:2], trans)
            detections[j, 2:4] = affine_transform(detections[j, 2:4], trans)

        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, src_image.shape[1])
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, src_image.shape[0])

        if visualize:
            # draw_annotations(src_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(src_image, detections[:5, :4], detections[:5, 4], detections[:5, 5].astype(np.int32),
                            label_to_name=generator.label_to_name,
                            score_threshold=score_threshold)

            # cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
            cv2.namedWindow('{}'.format(i), cv2.WINDOW_NORMAL)
            cv2.imshow('{}'.format(i), src_image)
            cv2.waitKey(0)

        # copy detections to all_detections
        for class_id in range(generator.num_classes()):
            all_detections[i][class_id] = detections[detections[:, -1] == class_id, :-1]

    return all_detections, total_time, img_count


def _get_annotations(generator):
    """
    Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_class_annotations, 5]

    Args:
        generator: The generator used to retrieve ground truth annotations.

    Returns:
        A list of lists containing the annotations for each image in the generator.

    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate_coco(
        generator,
        model,
        coco_true_path,
        det_json_path,
        model_file_name,
        score_threshold=0.01,
        max_detections=100,
        visualize=False,
        flip_test=False,
        keep_resolution=False
):
    """
    Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        max_detections: The maximum number of detections to use per image.
        visualize: Show the visualized detections or not.
        flip_test:

    Returns:
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, total_time, img_count = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     visualize=visualize, flip_test=flip_test, keep_resolution=keep_resolution)

    results = []
    # iterate through images
    for img_idx, image_detections in enumerate(all_detections, 1):

        # iterate through boxes
        for box_idx, box in enumerate(image_detections[0], 1):
            # append detection for each positively labeled class

            # change to (x, y, w, h) (MS COCO standard)
            box_coco = [0, 0, 0, 0]
            box_coco[0] = int(np.round(box[0]))
            box_coco[1] = int(np.round(box[1]))
            box_coco[2] = int(np.round(box[2] - box[0]))
            box_coco[3] = int(np.round(box[3] - box[1]))

            image_result = {
                'image_id': img_idx,
                'category_id': 0,
                'score': float(np.around(box[4], decimals=3)),
                'bbox': box_coco,
            }
            # append detection to results
            results.append(image_result)

    json.dump(results, open(os.path.join(det_json_path, 'etis_bbox_results_{}.json'.format(model_file_name)), 'w'),
              indent=4)

    # load results in COCO evaluation tool
    coco_true = COCO(coco_true_path)
    coco_pred = coco_true.loadRes(os.path.join(det_json_path, 'etis_bbox_results_{}.json'.format(model_file_name)))

    coco_eval = COCOeval(coco_true, coco_pred, iouType='bbox')
    coco_eval.params.imgIds = list(range(1, 197))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print()
    print("Total time:             ", total_time, " sec")
    print("Image count:            ", img_count)
    print("Average detection time: ", (total_time / img_count), " sec")

    return coco_eval.stats


def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple evaluation script only for ETIS dataset.')
    parser.add_argument('--csv_data_file', help='Path to CSV file containing annotations.')
    parser.add_argument('--csv_class_file', help='Path to a CSV file containing class label mapping.')
    parser.add_argument('--model_file', help='Path to the h5 model.')
    parser.add_argument('--coco_true', help='Path to JSON COCO ground truth.')
    parser.add_argument('--det_json_path', help='Folder in which the JSON with the detected boxes are saved.')
    parser.add_argument('--backbone', choices=['resnet50', 'resnet101', 'resnet152'], help='Chooses ResNet backbone')
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    from generators.csv_ import CSVGenerator
    from models.resnet import centernet
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test_generator = CSVGenerator(
        args.csv_data_file,
        args.csv_class_file,
        shuffle_groups=False,
    )
    model_path = args.model_file
    num_classes = test_generator.num_classes()
    flip_test = True
    nms = True
    keep_resolution = False
    score_threshold = 0.01
    model, prediction_model, debug_model = centernet(num_classes=num_classes,
                                                     nms=nms,
                                                     flip_test=flip_test,
                                                     freeze_bn=True,
                                                     score_threshold=score_threshold,
                                                     backbone=args.backbone)
    prediction_model.load_weights(model_path, by_name=True, skip_mismatch=True)

    model_file_name = args.model_file[args.model_file.rindex('/') + 1:]
    coco_eval_stats = evaluate_coco(test_generator,
                                    prediction_model,
                                    args.coco_true,
                                    args.det_json_path,
                                    model_file_name,
                                    visualize=False,
                                    flip_test=flip_test,
                                    keep_resolution=keep_resolution)


if __name__ == '__main__':
    main()
