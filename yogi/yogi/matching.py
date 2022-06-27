import glob
import os
import sys
import traceback

import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from skimage.color import hsv2rgb
from skimage.color import gray2rgb
from scipy import linalg, spatial


def colorize_frames(input_frames, template_frame, colorized_template, output_dir, image_bg=False, propagation_window=0, radius=10):
    """Uses SIFT matching to transfer color from a template frame to other frames.
    """
    
    if os.path.isfile(input_frames):
        frames = readlines(input_frames)
    else:
        assert(os.path.isdir(input_frames))
        frames = glob.glob(input_frames + '/*.png')
        frames = sorted(frames)

    rgb_template = plt.imread(fname=colorized_template)
    rgb_template = rgb_template[:,:,0:3]
    rgb_template = (255*rgb_template).astype(np.uint8)

    sift = cv2.SIFT_create(contrastThreshold=0.01)

    template_img = load_mono_image(template_frame)
    template_features = compute_features(template_img, sift)
    template_labels = np.arange(len(template_features[0]))

    try:
        os.makedirs(output_dir)
    except:
        pass

    frame_features = []
    frame_labels = []

    for i in range(len(frames)):

        if i % 10 == 0:
            print(i)

        img = load_mono_image(frames[i])
        features = compute_features(img, sift)
        frame_features.append(features)

        labels_from_template = propagate_labels(template_features, template_labels, frame_features[i])

        recent_frames = range(max(i - propagation_window, 0), i)

        labels_from_recent = []
        for j in recent_frames:
            propagated_labels = propagate_labels(frame_features[j], frame_labels[j], frame_features[i])
            labels_from_recent.append(propagated_labels)

        label_set = merge_label_sets([labels_from_template] + list(reversed(labels_from_recent)))

        frame_labels.append(label_set)

        if image_bg:
            bg_img = img.copy()
        else:
            bg_img = np.zeros_like(img)
            
        colorized_image = transfer_template_v2(template_features, features, label_set, rgb_template, bg_img, r=radius)
    
        new_fname = output_dir + '/' + frames[i].split('/')[-1]
        plt.imsave(fname=new_fname, arr=colorized_image)

        
def readlines(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines


def make_keypoint_masks(input_dir, output_dir, kp_size_thresh=20):
    """Create a mask where the locations of detected keypoints are 1
    """
    frames = glob.glob(input_dir + '/*.png')
    
    try:
        os.makedirs(output_dir)
    except:
        traceback.print_exc()
        sys.exit(1)
    
    sift = cv2.SIFT_create(contrastThreshold=0.01)
    for frame in frames:
        img = prep_for_sift(cv2.imread(frame)) # queryImage

        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        
        mask = mask_keypoints([k for k in kp if k.size < kp_size_thresh], img.shape[0:2])
        
        new_fname = output_dir + '/' + frame.split('/')[-1]
        plt.imsave(fname=new_fname, arr=mask, cmap=cm.gray)

        
def prep_for_sift(img, median_filt=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if median_filt:
        gray = median(gray)
    gray = (255 * gray.astype(np.float) / gray.max()).astype(np.uint8)
    return gray


def make_colorized_template(mask_files, saturation_direction='h', gradations=4):
    """Takes a list of 1-4 binary image mask files, and combines them into a single colorized template
    """
    assert(len(mask_files) > 0)
    assert(len(mask_files) <= 4)
    assert(saturation_direction in ['h','v'])
    
    mask_files = sorted(mask_files)
    colors = [
        (255 / 6) * 0 - 20,  # red
        (255 / 6) * 1 - 10,  # yellow
        (255 / 6) * 2 - 10,  # green
        (255 / 6) * 4 - 15   # blue
    ]
    masks = [cv2.imread(mask_fname)[:,:,0] > 0 for mask_fname in mask_files]

    h, w = masks[0].shape
    x, y = np.meshgrid(range(w), range(h))
    if saturation_direction == 'h':
        v = x
    else:
        v = y
    v = v.astype(np.float64)
    v = v / v.max()
    v = np.ceil(gradations * v) / gradations
    v = 255 * v
    v = v.reshape(v.shape + (1,)).astype(np.uint8)

    saturation = v
    value = np.zeros_like(saturation)
    hue = np.zeros_like(saturation)
    for mask, color in zip(masks, colors):
        hue[mask] = color
        value[mask] = 255

    hsv_template = np.concatenate((hue, saturation, value), axis=2)
    rgb_template = hsv2rgb(hsv_template)
    rgb_template = (rgb_template * 255).astype(np.uint8)

    return rgb_template


def transfer_template(matches, kp1, kp2, template, target_image):
    
    if len(target_image.shape) == 2:
        target_image = target_image.reshape(target_image.shape + (1,)).repeat(3, axis=2)
    else:
        target_image = target_image.copy()

    for m in matches:
        x2, y2 = kp2[m.trainIdx].pt
        x1, y1 = kp1[m.queryIdx].pt
        x1 = max(min(int(x1), template.shape[1] - 1), 0)
        y1 = max(min(int(y1), template.shape[0] - 1), 0)
        x2 = max(min(int(x2), template.shape[1] - 1), 0)
        y2 = max(min(int(y2), template.shape[0] - 1), 0)
        color = np.zeros((1, 3))
        color[0, :] = template[y1, x1, :] / 255.
        target_image[y2-10:y2+10, x2-10:x2+10, :] = color * 255

    return target_image


def transfer_template_v2(template_features, features, label_set, template_img, target_img, r=10):
    (template_kp, _) = template_features
    (kp, _) = features
    
    is_gray = len(target_img.shape) == 2
    if is_gray:
        target_img = gray2rgb(target_img)
    else:
        target_img = target_image.copy()
    
    for keypoint_idx in range(len(kp)):
        label = label_set[keypoint_idx]
        if label == -1:
            continue
        pt1 = template_kp[label].pt
        pt2 = kp[keypoint_idx].pt
        (x1, y1) = clip_img_idxs(pt1, template_img.shape)
        (x2, y2) = clip_img_idxs(pt2, target_img.shape)
        color = np.zeros((1, 3))
        color[0, :] = template_img[y1, x1, :] / 255.
        target_img[y2-r:y2+r, x2-r:x2+r, :] = color * 255

    return target_img


def clip_img_idxs(xy, shape):
    (x,y) = xy
    x = np.clip(x, 0, shape[1] - 1)
    y = np.clip(y, 0, shape[0] - 1)
    x = int(x)
    y = int(y)
    return (x,y)



def mask_keypoints(keypoints, shape, size=(10,10)):
    mask = np.zeros(shape)
    for kp in keypoints:
        x, y = kp.pt
        x, y = int(x), int(y)
        x = max(min(x, shape[1] - 1), 0)
        y = max(min(y, shape[0] - 1), 0)
        mask[y-10:y+10, x-10:x+10] = 1
    return mask


def get_coherent_neighbors(matches, train_keypoints, query_keypoints, epsilon=100, delta=100):

    point2s = [get_points(match, train_keypoints, query_keypoints)[1] for match in matches]
    if len(point2s) > 0:
        point2_tree = spatial.KDTree(point2s)

    coherent_neighbors = []
    
    for match in matches:    
        
        point1, point2 = get_points(match, train_keypoints, query_keypoints)

        # compute neighbors of point2
        point2_neighbors = point2_tree.query_ball_point(point2, epsilon)

        # lift neighbors of point2
        point2_neighbors_lifted = []
        for point2_neighbor in point2_neighbors:
            m = matches[point2_neighbor]
            p1, p2 = get_points(m, train_keypoints, query_keypoints)
            if p2 != point2:
                point2_neighbors_lifted.append(p1)
                
        if len(point2_neighbors_lifted) == 0:
            coherent_neighbors.append([])
            continue

        # compare lifted neighbors to point1
        point1 = np.array(point1)
        point2_neighbors_lifted = np.array(point2_neighbors_lifted)
        neighbor_diffs = linalg.norm(point2_neighbors_lifted - point1, axis=1)

        # apply a coherence test
        coherent_neighbors.append([])
        for idx in range(len(neighbor_diffs)):
            if neighbor_diffs[idx] < delta:
                coherent_neighbors[-1].append(point2_neighbors[idx])
    
    return coherent_neighbors


def get_points(match, train_keypoints, query_keypoints):
    train_point = train_keypoints[match.trainIdx].pt
    query_point = query_keypoints[match.queryIdx].pt
    return train_point, query_point


def merge_label_sets(label_sets):
    """Merge a list of label sets into one label set (sets earlier in the list get precedence)."""
    assert(len(label_sets) > 0)
    some_label_set = label_sets[0]
    merged_label_set = np.full_like(some_label_set, -1, dtype=np.int)
    for label_set in reversed(label_sets):
        valid_labels = label_set != -1
        merged_label_set[valid_labels] = label_set[valid_labels]
    return merged_label_set


def compute_features(img, sift):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return (keypoints, descriptors)


def propagate_labels(features1, labels1, features2):
    matches21 = compute_matches(features2, features1)
    labels2 = pullback(labels1, matches21, len(features2[0]))
    return labels2


def compute_matches(features1, features2, ratio_threshold=0.75, epsilon=50, delta=50, coherency=True):
    """Compute matches between sets of SIFT descriptors.
    
    Uses the ratio test and spatial coherency.
    """
    (kp1, des1) = features1
    (kp2, des2) = features2
    if len(kp1) == 0 or len(kp2) == 0:
        return []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for group in matches:
        if len(group) == 2:
            (m1, m2) = group
            if m1.distance < ratio_threshold * m2.distance:
                good_matches.append(m1)
    if coherency:
        coherent_neighbors = get_coherent_neighbors(good_matches, kp2, kp1, epsilon=epsilon, delta=delta)
        good_matches = [match for (match, neighbors) in zip(good_matches, coherent_neighbors) if len(neighbors) >= 2]
    return good_matches


def compute_incoherent_matches(features1, features2, ratio_threshold=0.75, epsilon=50, delta=50):
    """Compute matches between sets of SIFT descriptors.
    
    Uses the ratio test and spatial coherency.
    """
    (kp1, des1) = features1
    (kp2, des2) = features2
    if len(kp1) == 0 or len(kp2) == 0:
        return []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for group in matches:
        if len(group) == 2:
            (m1, m2) = group
            if m1.distance < ratio_threshold * m2.distance:
                good_matches.append(m1)
    coherent_neighbors = get_coherent_neighbors(good_matches, kp2, kp1, epsilon=epsilon, delta=delta)
    bad_matches = [match for (match, neighbors) in zip(good_matches, coherent_neighbors) if len(neighbors) <= 1]
    return bad_matches


def pullback(labels1, matches21, nfeatures2):
    labels2 = np.full(nfeatures2, -1, dtype=np.int)
    for match21 in matches21:
        labels2[match21.queryIdx] = labels1[match21.trainIdx]
    return labels2


def load_mono_image(fname):
    return prep_for_sift(cv2.imread(fname))
