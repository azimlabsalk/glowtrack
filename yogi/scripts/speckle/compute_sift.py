import glob
import multiprocessing
from multiprocessing import Pool
import os
import sys

import cv2
import numpy as np


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


def compute_and_save_sift(fname):
    # load image
    img = load_image(fname)
    # compute sift features
    (kp, des) = compute_sift(img)
    if des is None:
        kp = []
        des = np.array([[]])
    # serialize features
    s = stringify_sift_list(kp, des)
    # save to a '.key' or '.key.gz' file
    output_fname = fname.replace('png', 'key')
    with open(output_fname, 'w') as f:
        f.write(s)


if __name__ == '__main__':

    # image_list_file = sys.argv[1] 
    # image_list = readlines(image_list_file)

    image_dir = sys.argv[1] 
    image_list = glob.glob(os.path.join(image_dir, '*.png'))

    n_cpus = multiprocessing.cpu_count()    

    with Pool(n_cpus) as p:
        p.map(compute_and_save_sift, image_list)


