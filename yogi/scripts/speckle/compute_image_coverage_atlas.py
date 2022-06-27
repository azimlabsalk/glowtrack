import glob
import multiprocessing
from multiprocessing import Pool
import os
import sys

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from yogi.matching import compute_matches


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


def render_atlas(image, image_features, covering_images, covering_features, matched_idxs):

    k = 20
    n = len(covering_images)
    columns = 5
    rows = int(np.ceil((n + k) / columns))

    plt.figure(figsize=(10, 12))

    # render main image
    plt.subplot(rows, columns, (1, k))
    img = load_image(image)
    plt.imshow(img, cmap='gray')
    (kp, _) = image_features
    for (rank, (image_idxs, _)) in enumerate(matched_idxs):
        # render main image features
        label = f'image {rank}'
        zorder = 1 / (1 + rank)
        render_keypoints([kp[idx] for idx in image_idxs], s=15, label=label, zorder=zorder)
    plt.axis('off')
    plt.legend(loc='lower left')

    # render covering images
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    sub_idx = k + 1
    for (rank, (covering_image, (kp, _), (_, covering_idxs))) in enumerate(zip(covering_images, covering_features, matched_idxs)):
        plt.subplot(rows, columns, sub_idx)
        sub_idx += 1
        img = load_image(covering_image)
        plt.imshow(img, cmap='gray')
        render_keypoints([kp[idx] for idx in covering_idxs], s=15, label=label, zorder=zorder, color=colors[rank])
        plt.axis('off')


def compute_and_save_covering(args):
    (image, covering_images, output_image) = args
    # load image and sift features
    img = load_image(image)
    image_features = sift_list_from_file(image.replace('png', 'key')) 
    # load covering images sift features
    covering_features = [sift_list_from_file(f.replace('png', 'key')) for f in covering_images]
    # compute matched features
    matched_idxs = [get_covered_idxs(image_features, features) for features in covering_features]
    # render covering matches on image
    render_atlas(image, image_features, covering_images, covering_features, matched_idxs)
    # save image
    plt.savefig(output_image)
    plt.close('all')


if __name__ == '__main__':

    # image_list_file = sys.argv[1] 
    # image_list = readlines(image_list_file)

    image_dir = sys.argv[1] 
    covering_file = sys.argv[2] 
    output_dir = sys.argv[3]

    image_list = glob.glob(os.path.join(image_dir, '*.png'))
    with open(covering_file, 'r') as f:
        covering_images = [line.strip() for line in f.readlines()]

    args = []
    for image in image_list:
        basename = os.path.basename(image)
        output_image = os.path.join(output_dir, basename)
        args.append((image, covering_images, output_image))

    n_cpus = multiprocessing.cpu_count()    

    with Pool(n_cpus) as p:
        p.map(compute_and_save_covering, args)


