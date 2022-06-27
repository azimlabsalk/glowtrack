import multiprocessing
from multiprocessing import get_context
from multiprocessing import Pool
import os
import glob
import sys

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import click
from yogi.matching import compute_matches
from yogi.db import session
from yogi.models import *


def readlines(fname):
    with open(fname, 'r') as f:
        lines = [x.strip() for x in f.readlines()]
        return lines


def compute_sift(img, contrastThreshold=0.01):
    sift = cv2.SIFT_create(contrastThreshold=contrastThreshold)
    kp, des = sift.detectAndCompute(img, None)
    return (kp, des)


def load_image(fname):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (255 * gray.astype(np.float) / gray.max()).astype(np.uint8)
    return gray

def unstringify_sift(s, from_radians=True):
    lines = s.split('\n')
    (y, x, size, angle) = lines[0].split(' ')
    y = float(y)
    x = float(x)
    size = float(size)
    angle = float(angle)
    
    if from_radians:
        angle = np.rad2deg(angle + np.pi)
    
    des_string = ''.join(lines[1:])
    des_string.replace('\n','')
    des_entries = des_string.strip().split(' ')
    des_entries = [int(e) for e in des_entries]
    des = np.array(des_entries, dtype=np.float32)
    
    kp = cv2.KeyPoint()
    kp.pt = (x, y)
    kp.size = size
    kp.angle = angle

    return (kp, des)

def unstringify_sift_list(s):
    lines = s.strip().split('\n')
    n_features = int(lines[0].split(' ')[0])
    
    keypoints = []
    descriptors = []
    for i in range(1, len(lines), 8):
        des_string = '\n'.join(lines[i:i+8])
        (kp, des) = unstringify_sift(des_string)
        keypoints.append(kp)
        descriptors.append(des)
    
    return (keypoints, np.array(descriptors))


def sift_list_from_file(fname):
    with open(fname, 'r') as f:
        s = f.read().strip()
    return unstringify_sift_list(s)


def stringify_sift_list(keypoints, descriptors):
    assert(descriptors is not None)

    s = ''
    
    # header
    num_kps = len(keypoints)
    desc_length = 128
    s += f'{num_kps} {desc_length}\n'
    
    strings = []
    for (kp, des) in zip(keypoints, descriptors):
        strings.append(stringify_sift(kp, des))
    s += '\n'.join(strings)

    return s
        
def stringify_sift(keypoint, descriptor, use_radians=True):
    (x, y) = keypoint.pt
    size = keypoint.size
    angle = keypoint.angle
    if use_radians:
        angle = np.deg2rad(angle) - np.pi
    kp_string = '{:.2f} {:.2f} {:.2f} {:.3f}'.format(y, x, size, angle)
    
    des_string = ''
    entries = list(descriptor.astype(np.uint8))
    entries = list(map(str, entries))
    for i in range(7):
        des_string += '\n '
        des_string += ' '.join(entries[i*20:(i+1)*20])
    
    s = kp_string + des_string
    return s


def get_covered_idxs(image_features, covering_features, template2image=True):
    if template2image:
        matches = compute_matches(covering_features,
                                      image_features,
                                      ratio_threshold=0.75,
                                      epsilon=50,
                                      delta=50)
        image_idxs = [match.trainIdx for match in matches]
        template_idxs = [match.queryIdx for match in matches]
    else:
        matches = compute_matches(image_features,
                                      covering_features,
                                      ratio_threshold=0.75,
                                      epsilon=50,
                                      delta=50)
        image_idxs = [match.queryIdx for match in matches]
        template_idxs = [match.trainIdx for match in matches]
    return (image_idxs, template_idxs)


def render_keypoints(kps, **kwargs):

    x_lst = []
    y_lst = []

    for kp in kps:
        (x, y) = kp.pt
        x_lst.append(x)
        y_lst.append(y)

    plt.scatter(x_lst, y_lst, **kwargs)


def segment_contains(segment, pt):
    (x, y) = pt
    (h, w) = segment.shape
    x = min(max(int(x), 0), w - 1)
    y = min(max(int(y), 0), h - 1)
    return segment[y, x]


def compute_labels(image, image_features, covering_images, covering_features, matched_idxs, templates, landmarkset_name):

    k = 20
    n = len(covering_images)
    columns = 5
    rows = int(np.ceil((n + k) / columns))
    n_segments = len(templates[0])

    (kp, _) = image_features

    # load landmarkset
    print('loading landmarkset')
    landmarkset = session.query(LandmarkSet).filter_by(name=landmarkset_name).one()
    landmark_ids = [l.id for l in landmarkset.landmarks] 

    # load image_id
    visible_image = image.replace('uv', 'visible')
    print('loading image_id for path {}'.format(visible_image))
    image_id = session.query(Image).filter_by(path=visible_image).one().id
    (h, w) = plt.imread(visible_image).shape[0:2]

    # group together all the points for one segment (all templates)
    print('grouping points by segment')
    segment_points = [[] for _ in range(n_segments)]
    for (template, (image_idxs, covering_idxs), (kp_, _)) in zip(templates, matched_idxs, covering_features):
        template_kp = [(kp[idx], kp_[idx_]) for (idx, idx_) in zip(image_idxs, covering_idxs)]
        for i in range(n_segments):
            segment = template[i]
            segment_kp = [k for (k, k_) in template_kp if segment_contains(segment, k_.pt)]
            segment_points[i].extend(segment_kp)

    labels = []
    print('computing label coordinates')
    # compute label coordinates 
    for (i, (points, landmark_id)) in enumerate(zip(segment_points, landmark_ids)):
        label = Label(landmark_id=landmark_id, image_id=image_id)
        if len(points) >= 5:
            x = np.median([p.pt[0] for p in points])
            y = np.median([p.pt[1] for p in points])
            label.x = x / w
            label.y = y / h
        else:
            assert(label.is_hidden())
        labels.append(label)

    return labels


def load_templates(template_dir):
    templates = []
    image_dirs = sorted(glob.glob(template_dir + '/*'))
    for image_dir in image_dirs:
        templates.append([])
        segment_files = sorted(glob.glob(image_dir + '/*.png'))
        for segment_file in segment_files:
            image = plt.imread(segment_file)
            mask = image[:, :, 3] > 0
            mask = mask.reshape(mask.shape[0:2])
            templates[-1].append(mask)
    return templates

import skimage
from skimage.io import imread, imsave
from skimage import morphology

def clean_up_mask(mask, iters=5):

    mask_ = morphology.remove_small_objects(mask)
    
    for _ in range(iters):
        mask_ = morphology.binary_dilation(mask_)

    for _ in range(iters):
        mask_ = morphology.binary_erosion(mask_)

    return mask_

def uv_to_mask(uv_image):
    return uv_image[:, :, 0] > 0


def compute_and_save_mask(args):
    (uv_image_path, output_dir) = args
    # load image
    uv_image = imread(uv_image_path)
    # compute mask
    iters = 0
    uv_mask = uv_to_mask(uv_image)
    uv_mask_cleaned = clean_up_mask(uv_mask, iters=iters)
    # save mask
    (_, fname) = os.path.split(uv_image_path)
    outfile = os.path.join(output_dir, fname)
    imsave(outfile, uv_mask_cleaned.astype(np.uint8))


@click.command('compute_masks', no_args_is_help=True)
@click.argument('image_dir')
@click.argument('output_dir')
def main(image_dir, output_dir): 
    """compute_labels
    
    \b
    IMAGE_DIR - contains uv speckle images in PNG format
    OUTPUT_DIR - contains mask images in PNG format
    """

    image_list = glob.glob(os.path.join(image_dir, '*.png'))

    os.makedirs(output_dir, exist_ok=True)

    args = []
    for image in image_list:
        args.append((image, output_dir))

    n_cpus = multiprocessing.cpu_count()    

    with get_context("spawn").Pool(n_cpus) as p:
        p.map(compute_and_save_mask, args)


if __name__ == '__main__':
    sys.exit(main())

