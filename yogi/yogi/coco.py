"""Tools for export and import of data."""

import datetime


def coco_gt_image(image):
    data = {
        "id": image.id,
        "width": image.width,
        "height": image.height,
        "file_name": image.path,
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

