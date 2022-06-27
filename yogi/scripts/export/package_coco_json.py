import json
import os
import shutil
import sys


def transform_file_name(file_name, image_root_dir, output_dir):
    assert(file_name[:len(image_root_dir)] == image_root_dir)
    new_file_name = file_name[len(image_root_dir):]
    new_file_name = os.path.join(output_dir, new_file_name)
    return new_file_name


if __name__ == "__main__":
    print(sys.argv)

    coco_json = sys.argv[1]
    image_root_dir = sys.argv[2]  # this must be a common ancestor of all image paths
    if image_root_dir[-1] != '/':
        image_root_dir = image_root_dir + '/'
    output_dir = sys.argv[3]
    dataset_name = sys.argv[4]

    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, '{}.json'.format(dataset_name))

    with open(coco_json, 'r') as f:
        data = json.load(f)
        for image in data['images']:
            file_name = image['file_name']
            new_file_name = transform_file_name(file_name, image_root_dir, output_dir)
            print('file in: {}, file out: {}'.format(file_name, new_file_name))

