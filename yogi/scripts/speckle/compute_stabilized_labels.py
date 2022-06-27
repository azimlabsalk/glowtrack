from collections import Counter
import multiprocessing
from multiprocessing import get_context
from multiprocessing import Pool
import os
import pickle
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


def eliminate_multimatches(matches):
    counts1 = Counter([m.queryIdx for m in matches])
    dups1 = set([x for (x, c) in counts1.items() if c > 1])
    counts2 = Counter([m.trainIdx for m in matches])
    dups2 = set([x for (x, c) in counts2.items() if c > 1])
    matches = [m for m in matches if m.queryIdx not in dups1 and m.trainIdx not in dups2]
    return matches


def get_covered_idxs(image_features, covering_features, template2image=True):
    if template2image:
        matches = compute_matches(covering_features,
                                      image_features,
                                      ratio_threshold=0.75,
                                      epsilon=50,
                                      delta=50)
        matches = eliminate_multimatches(matches)
        image_idxs = [match.trainIdx for match in matches]
        template_idxs = [match.queryIdx for match in matches]
    else:
        matches = compute_matches(image_features,
                                      covering_features,
                                      ratio_threshold=0.75,
                                      epsilon=50,
                                      delta=50)
        matches = eliminate_multimatches(matches)
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


def has_labels(image, labelsource_name):
    # load image_id
    visible_image = image.replace('uv', 'visible')
    print('loading image_id for path {}'.format(visible_image))
    image_id = session.query(Image).filter_by(path=visible_image).one().id

    n_existing_labels = session.query(Label).join(LabelSource).filter(LabelSource.name == labelsource_name).filter(Label.image_id == image_id).count()
    return n_existing_labels > 0


def create_array(keypoints, idxs):
    pts = [keypoints[idx].pt for idx in idxs]
    pos = np.array(pts)
    pos = pos.astype(np.float32)
    return pos


def compute_stabilized_points(node_orig_idxs, match_idxs1, match_idxs2, kp1, kp2, node_neighbors, stabilize=True, min_matches=10):
    """For each node in img1, compute its stabilized position in img2"""

    node_homs = []

    # node_orig_idxs = [top_features[i] for i in warp_node_idxs]
    matched_node_idxs = list(set(node_orig_idxs).intersection(set(match_idxs1)))
    matched_node_idxs = set(matched_node_idxs)

    match_map = {idx1: idx2 for (idx1, idx2) in zip(match_idxs1, match_idxs2)}

    all_pts1 = create_array(kp1, range(len(kp1))) 
    all_pts2 = create_array(kp2, range(len(kp2)))

    warp_nodes = all_pts1[node_orig_idxs, :]

    mapped_pts = []
    for i in range(len(node_neighbors)):

        neighbor_matches1 = list(set(match_idxs1).intersection(node_neighbors[i]))
        neighbor_matches2 = [match_map[idx1] for idx1 in neighbor_matches1]

        matched_neighbor_pts1 = all_pts1[neighbor_matches1, :]
        matched_neighbor_pts2 = all_pts2[neighbor_matches2, :]

        # homography

        if node_orig_idxs[i] in matched_node_idxs:

            idx2 = match_map[node_orig_idxs[i]]
            pt = kp2[idx2].pt

        elif matched_neighbor_pts1.shape[0] >= min_matches and stabilize:

            # M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 50)
            # M, mask = cv2.findHomography(matched_neighbor_pts1, matched_neighbor_pts2, cv2.RANSAC, 50)  # uses all the points

            M, mask = cv2.findHomography(matched_neighbor_pts1, matched_neighbor_pts2)  # uses all the points
            mask = mask.flatten().astype(np.bool)

            # pts1_transformed = cv2.perspectiveTransform(np.array([pts1]), M)
            node_transformed = perspectiveTransform(warp_nodes[i:i+1, :], M)
            node_transformed = node_transformed[0]
            pt = (node_transformed[0, 0], node_transformed[0, 1])

        else:

            pt = None

        mapped_pts.append(pt)

    return mapped_pts


def perspectiveTransform(p, A):
#     p = p[0]
    p_aug = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)
    o_aug = A.dot(p_aug.T).T
    o = o_aug[:, 0:2] / np.expand_dims(o_aug[:, 2], axis=1)
    return np.array([o])


def compute_labels(image, image_features, template_features, matched_idxs, dict_to_track, landmarkset_name, labelsource_name, stabilize, min_matches):

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

    n_existing_labels = session.query(Label).join(LabelSource).filter(LabelSource.name == labelsource_name).filter(Label.image_id == image_id).count()
    if n_existing_labels > 0:
        return []

    # compute labels
    with open(dict_to_track, 'rb') as f:
        data_dict = pickle.load(f)
        good_idxs = data_dict['good_idxs']
        warp_node_idxs = data_dict['warp_node_idxs']
        node_orig_idxs = data_dict['node_orig_idxs']
        node_neighbors = data_dict['node_neighbors']

    assert(len(node_orig_idxs) == len(landmarkset.landmarks))
    assert(len(node_orig_idxs) == len(warp_node_idxs))

    (image_idxs, template_idxs) = matched_idxs
    mapped_pts = compute_stabilized_points(node_orig_idxs, template_idxs, image_idxs, template_features[0], image_features[0], node_neighbors, stabilize=stabilize, min_matches=min_matches)

    # template_to_image = {idx1: idx2 for (idx1, idx2) in zip(template_idxs, image_idxs)}

    print('computing label coordinates')
    labels = []
    for (pt, landmark_id) in zip(mapped_pts, landmark_ids):
        label = Label(landmark_id=landmark_id, image_id=image_id)
        if pt is not None:
            (x, y) = pt
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


def compute_and_save_labels(args):
    (image, template_uv_image, dict_to_track, landmarkset_name, labelset_name, labelsource_name, stabilize, min_matches) = args
    # load image and sift features
    print('load image and sift features')
    image_features = sift_list_from_file(image.replace('png', 'key'))
    # load covering images sift features
    print('load template sift features')
    template_features = sift_list_from_file(template_uv_image.replace('png', 'key'))
    # compute matched features
    print('compute matched features')
    matched_idxs = get_covered_idxs(image_features, template_features)
    # compute labels 
    print('computing labels')
    labels = compute_labels(image, image_features, template_features, matched_idxs, dict_to_track, landmarkset_name, labelsource_name, stabilize, min_matches)
    # set the label source
    print('set the label source')
    source = session.query(LabelSource).filter_by(name=labelsource_name).one()
    for label in labels:
        label.source_id = source.id
        session.add(label)
    # save labels
    print('save {} labels'.format(len(labels)))
    # The following was agonizingly slow, so do a batch bulk insert at the end
    #labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    #labelset.labels.extend(labels)
    #session.add(labelset)
    session.commit()


@click.command('compute_labels', no_args_is_help=True)
@click.argument('image_dir')
@click.argument('template_uv_image')
@click.argument('dict_to_track')
@click.argument('landmarkset_name')
@click.argument('labelset_name')
@click.argument('labelsource_name')
@click.argument('stabilize', type=bool)
@click.argument('min_matches', type=int, default=10)
def main(image_dir, template_uv_image, dict_to_track, landmarkset_name, labelset_name, labelsource_name, stabilize, min_matches):
    """compute_labels
    
    \b
    IMAGE_DIR - contains uv speckle images in PNG format
    \b
    TEMPLATE_UV_IMAGE - uv speckle image
    \b
    DICT_TO_TRACK - pickled dictionary with nodes to track
    \b
    LANDMARKSET_NAME - name of landmarkset, one landmark per template segment
    \b
    LABELSET_NAME - name of labelset to store labels in
    \b
    LABELSOURCE_NAME - name of labelsource to attribute labels to
    \b
    MIN_MATCHES - the minimum number of neighbors for a homography
    """

    image_list = glob.glob(os.path.join(image_dir, '*.png'))

    print('Found {} images in {}'.format(len(image_list), image_dir))

    # create labelset if it doesn't already exist
    if session.query(LabelSet).filter_by(name=labelset_name).count() == 0:
        labelset = LabelSet(name=labelset_name)
        session.add(labelset)
        session.commit()

    args = []
    for image in image_list:
        if not has_labels(image, labelsource_name):
            args.append((image, template_uv_image, dict_to_track, landmarkset_name, labelset_name, labelsource_name, stabilize, min_matches))

    n_cpus = 30 #multiprocessing.cpu_count() 

    with get_context("spawn").Pool(n_cpus) as p:
        p.map(compute_and_save_labels, args)

    labelset_id = session.query(LabelSet).filter_by(name=labelset_name).one().id
    label_ids = session.query(Label.id).join(LabelSource).filter(LabelSource.name == labelsource_name).all()
    value_dicts = [{'labelset_id': labelset_id, 'label_id': label_id} for label_id in label_ids]
    session.connection().execute(labelset_association_table.insert(values=value_dicts))
    session.commit()


if __name__ == '__main__':
    sys.exit(main())

