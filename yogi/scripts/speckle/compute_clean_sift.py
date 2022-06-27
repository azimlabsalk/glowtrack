import glob
import multiprocessing
from multiprocessing import Pool
import os
import sys

import cv2
import numpy as np

from yogi.matching import compute_matches


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


def readlines(fname):
    with open(fname, 'r') as f:
        lines = [x.strip() for x in f.readlines()]
        return lines


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


def get_frame_num(fname):
    return int(os.path.splitext(os.path.basename(fname))[0])


def replace_frame_num(fname, i, format_length=8):
    (d, f) = os.path.split(fname)
    (_, ext) = os.path.splitext(f)
    new_f = '{:08d}'.format(i) + ext
    new_fname = os.path.join(d, new_f)
    return new_fname


def compute_and_save_clean_sift(fname):
    # compute adjacent fname
    i = get_frame_num(fname)
    adjacent_fnames = [
        replace_frame_num(fname, i - 1),
        replace_frame_num(fname, i + 1)
    ]
    for f in adjacent_fnames + [fname]:
        if not os.path.exists(f):
            return
    # load key files
    (kp, des) = sift_list_from_file(fname)
    selected_idxs = []
    for adjacent_fname in adjacent_fnames:
        (kp_, des_) = sift_list_from_file(adjacent_fname)
        # perform matching
        good_matches = compute_matches((kp_, des_),
                                       (kp, des),
                                       ratio_threshold=0.75,
                                       epsilon=50,
                                       delta=50)
        # select features that match any adjacent frame
        selected_idxs.extend([match.trainIdx for match in good_matches])
    selected_idxs = list(set(selected_idxs))
    kp = [kp[i] for i in selected_idxs]
    des = [des[i] for i in selected_idxs]
    # serialize features
    s = stringify_sift_list(kp, des)
    # save to a '.key' or '.key.gz' file
    output_fname = fname.replace('key', 'key-clean')
    with open(output_fname, 'w') as f:
        f.write(s)


if __name__ == '__main__':

    # image_list_file = sys.argv[1] 
    # image_list = readlines(image_list_file)

    key_dir = sys.argv[1] 
    key_list = glob.glob(os.path.join(key_dir, '*.key'))

    n_cpus = multiprocessing.cpu_count()    

    with Pool(n_cpus) as p:
        p.map(compute_and_save_clean_sift, key_list)


