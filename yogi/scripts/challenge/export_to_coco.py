"""Tools for export and import of data."""

import datetime
import json
import os
import shutil
import sys

from yogi.db import session
from yogi.models import *
from yogi.sql import get_labels


def coco_gt_image(image, file_name=None, flip=False):
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


def coco_gt_annotation(label, flip=False):

    if len(label.image.bounding_boxes) > 0:
        bb = label.image.bounding_boxes[0]
        if flip and bb.x is not None:
            img_width = label.image.width
            bbox = [img_width - bb.x, bb.y, bb.width, bb.height]
        else:
            bbox = [bb.x, bb.y, bb.width, bb.height]
        area = bb.width * bb.height if (bb.width is not None and bb.height is not None) else None
    else:
        bbox = []
        bbox = [-1, -1, -1, -1]
        area = -1

    visibility = 0 if label.is_hidden() else 2
    if flip and not label.is_hidden():
        img_width = label.image.width
        keypoints = [img_width - label.x_px, label.y_px, visibility]
    else:
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


def copy_image(old_img, new_img, flip=False):
    from numpy import fliplr
    from skimage.io import imread, imsave

    if flip:
        arr = imread(old_img)
        arr = fliplr(arr)
        imsave(new_img, arr)
    else:
        shutil.copy(old_img, new_img)


def export_subclipset(subclipset_name, labelset_name, landmark_id, output_json, clips_dir, copy_images=True,
                      flip_images=False):

    scs = session.query(SubClipSet).filter_by(name=subclipset_name).one()

    print('exporting ' + scs.name + ' to ' + output_dir)

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

    for (i, sc) in enumerate(sorted(scs.subclips, key=lambda sc: sc.id)):

        clip_output_dir = '{}/clip{:03d}'.format(clips_dir, i)
        print(clip_output_dir)

        labels = get_labels(labelset_name=labelset_name, subclip_id=sc.id, landmark_id=landmark_id)
        print(len(labels))
        imageid2label = {label.image.id: label for label in labels}

        landmark_ids = list(set([label.landmark.id for label in labels]))
        assert(len(landmark_ids) == 1)
        landmark_name = labels[0].landmark.name

        os.makedirs(clip_output_dir, exist_ok=True)

        for (j, image) in enumerate(sorted(sc.images, key=lambda image: image.frame_num)):

            label = imageid2label.get(image.id, None)
            new_path = os.path.join(clip_output_dir, '{:08d}.jpg'.format(j))
            print(new_path)
            if copy_images:
                copy_image(image.path, new_path, flip=flip_images)

            image_json = coco_gt_image(image, file_name=new_path, flip=flip_images)
            images.append(image_json)

            if label:
                annotation = coco_gt_annotation(label, flip=flip_images)

                visibility = annotation['keypoints'][2]
                
                if visibility == 0:
                    annotation['keypoints'][0] = 0
                    annotation['keypoints'][1] = 0
                
                if annotation['area'] is None:
                    annotation['area'] = 0
                    annotation['bbox'] = [0, 0, 0, 0]

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

    subclipset_name = 'challenge-2-left-with-flipped'
    labelset_name = 'challenge-2-left-with-flipped'
    landmark_id = 1

    export_subclipset(subclipset_name, labelset_name, landmark_id, "dataset.json", clips_dir,
                      copy_images=True, flip_images=True)
