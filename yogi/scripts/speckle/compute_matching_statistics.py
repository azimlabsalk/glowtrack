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


def compute_stats(matched_idxs):
    # one pair of matched idxs for each covering image
    covered_idxs = set()
    stats = []
    for (image_idxs, covering_idxs) in matched_idxs:
        covered_idxs = covered_idxs.union(image_idxs)
        stats.append(len(covered_idxs))

    return stats


def compute_and_save_match_stats(args):
    (image, covering_images) = args
    # load image and sift features
    print('load image and sift features')
    image_features = sift_list_from_file(image.replace('png', 'key'))
    # load covering images sift features
    print('load covering images sift features')
    covering_features = [sift_list_from_file(f.replace('png', 'key')) for f in covering_images]
    # compute matched features
    print('compute matched features')
    matched_idxs = [get_covered_idxs(image_features, features) for features in covering_features]
    # compute stats
    stats = compute_stats(matched_idxs)
    # save the stats
    s = ' '.join(map(str, stats))
    stats_file = image.replace('png', 'match-stats-top10.txt')
    print(f'saving stats for {image}: {s}')
    with open(stats_file, 'w') as f:
        f.write(s)


@click.command('compute_match_stats', no_args_is_help=True)
@click.argument('image_dir')
@click.argument('covering_file')
def main(image_dir, covering_file):
    """compute_match_stats
    \b
    IMAGE_DIR - contains uv speckle images in PNG format
    \b
    COVERING_FILE - contains a list of uv speckle image paths to be used as covering templates
    """

    image_list = glob.glob(os.path.join(image_dir, '*.png'))
    with open(covering_file, 'r') as f:
        covering_images = [line.strip() for line in f.readlines()]

    print('Found {} images in {}'.format(len(image_list), image_dir))

    args = []
    for image in image_list:
        args.append((image, covering_images))

    n_cpus = multiprocessing.cpu_count() 

    with get_context("spawn").Pool(n_cpus) as p:
        p.map(compute_and_save_match_stats, args)


if __name__ == '__main__':
    sys.exit(main())

