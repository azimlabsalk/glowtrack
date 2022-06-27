"""Tools for export and import of data."""

import datetime
import json
import os
import shutil
import sys

from yogi.db import session
from yogi.models import *
from yogi.sql import get_labels


def coco_gt_image(image, file_name=None):
    data = {
        "id": image.id,
        "width": image.width,
        "height": image.height,
        "file_name": file_name if file_name else image.path,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": str(datetime.datetime.now()),
    }
    return data


def coco_gt_annotation(label):

    if len(label.image.bounding_boxes) > 0:
        bb = label.image.bounding_boxes[0]
        bbox = [bb.x, bb.y, bb.width, bb.height]
        area = bb.width * bb.height if (bb.width is not None and bb.height is not None) else None
    else:
        bbox = []
        bbox = [-1, -1, -1, -1]
        area = -1

    visibility = 0 if label.is_hidden() else 2
    keypoints = [label.x_px, label.y_px, visibility]

    annotation = {
        "id": label.id,
        "image_id": label.image_id,
        "category_id": 1,
        "segmentation": [],
        "area": area,
        "bbox": bbox, 
        "iscrowd": 0,
        "keypoints": keypoints,
        "num_keypoints": 1,
    }

    return annotation


def export_coco_gt(labels):
    json_data = {}

    now = datetime.datetime.now()
    info = {
        "year": now.year,
        "version": "",
        "description": "",
        "contributor": "",
        "url": "",
        "date_created": str(now), 
    }

    images = []
    annotations = []

    for (i, label) in enumerate(labels):

        if i % 1000 == 0:
            print('exporting label {}'.format(i))

        image = coco_gt_image(label.image)
        annotation = coco_gt_annotation(label)

        visibility = annotation['keypoints'][2]
        
        if visibility == 0:
            annotation['keypoints'][0] = 0
            annotation['keypoints'][1] = 0
        
        if annotation['area'] is None:
            annotation['area'] = 0
            annotation['bbox'] = [0, 0, 0, 0]
 
        images.append(image)
        annotations.append(annotation)

        #if annotation['area'] > 0:
        #    images.append(image)
        #    annotations.append(annotation)

    landmark_ids = set([label.landmark.id for label in labels])

    assert(len(landmark_ids) == 1)

    landmark_name = labels[0].landmark.name

    json_data['info'] = info
    json_data['images'] = images
    json_data['annotations'] = annotations
    json_data['licenses'] = []
    json_data['categories'] = [{'supercategory': 'object', 'name': 'object', 'skeleton': [], 'keypoints': [landmark_name], 'id': 1}]

    return json_data


def coco_result(label):
    result = {
        "image_id": label.image_id,
        "category_id": 1,
        "keypoints": [label.x_px, label.y_px, 1],
        "score": label.confidence,
    }
    return result


def export_coco_results(labels):

    results = []

    for label in labels:
        result = coco_result(label)
        results.append(result)

    return results


def export_clipgroupset(clipgroupset_name, labelset_name, output_json, clips_dir, copy_images=True):
    cgs = session.query(ClipGroupSet).filter_by(name=clipgroupset_name).one()

    print('exporting ' + cgs.name + ' to ' + output_dir)

    json_data = {}

    now = datetime.datetime.now()
    info = {
        "year": now.year,
        "version": "",
        "description": "",
        "contributor": "",
        "url": "",
        "date_created": str(now), 
    }

    images = []
    annotations = []
    landmark_ids = []

    for cg in sorted(cgs.clipgroups, key=lambda cg: cg.id):

        cg_output_dir = '{}/clip{:03d}'.format(clips_dir, cg.id)
        print(cg_output_dir)

        for clip in cg.clips:
    
            clip_output_dir = '{}/cam{}'.format(cg_output_dir, clip.camera_index)
            print(clip_output_dir)

            labels = get_labels(labelset_name=labelset_name, clip_id=clip.id)
            print(len(labels))

            landmark_ids = list(set([label.landmark.id for label in labels]))
            assert(len(landmark_ids) == 1)
            landmark_name = labels[0].landmark.name

            os.makedirs(clip_output_dir, exist_ok=True)

            for label in labels:

                image = label.image
                new_path = os.path.join(clip_output_dir, os.path.basename(image.path))
                print(new_path)
                if copy_images:
                    shutil.copy(image.path, new_path)

                image_json = coco_gt_image(image, file_name=new_path)
                annotation = coco_gt_annotation(label)

                visibility = annotation['keypoints'][2]
                
                if visibility == 0:
                    annotation['keypoints'][0] = 0
                    annotation['keypoints'][1] = 0
                
                if annotation['area'] is None:
                    annotation['area'] = 0
                    annotation['bbox'] = [0, 0, 0, 0]
         
                images.append(image_json)
                annotations.append(annotation)

    json_data['info'] = info
    json_data['images'] = images
    json_data['annotations'] = annotations
    json_data['licenses'] = []
    json_data['categories'] = [{'supercategory': 'object', 'name': 'object', 'skeleton': [], 'keypoints': [landmark_name], 'id': 1}]

    with open(output_json, 'w') as f:
        json.dump(json_data, f)


if __name__ == '__main__':

    output_dir = sys.argv[1]

    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    os.chdir(output_dir)

    clips_dir = 'clips'

    clipgroupset_name = 'cerebro-all-behaviors-clean'
    labelset_name = 'cerebro-all-behaviors-clean'

    export_clipgroupset(clipgroupset_name, labelset_name, "dataset.json", clips_dir, copy_images=True)
    export_clipgroupset(clipgroupset_name + '-test', labelset_name, "test.json", clips_dir, copy_images=False)
    export_clipgroupset(clipgroupset_name + '-train', labelset_name, "train.json", clips_dir, copy_images=False)



