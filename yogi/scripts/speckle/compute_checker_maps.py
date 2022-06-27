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


def checkboard_like(arr, sq=10):
    h_stripes = np.zeros_like(arr, dtype=np.bool)
    x = 0
    while x < h_stripes.shape[1]:
        h_stripes[:, x:x+sq, :] = (x / sq) % 2 == 0
        x += sq
    v_stripes = np.ones_like(arr)
    y = 0
    while y < v_stripes.shape[0]:
        v_stripes[y:y+sq, :, :] = (y / sq) % 2 == 0
        y += sq
    checkerboard = np.logical_xor(h_stripes, v_stripes)
    return checkerboard


def compute_flow(dims, image_features, template_features, image_matched_idxs, template_matched_idxs):
    from scipy.interpolate import griddata
    from skimage.color import hsv2rgb

    h, w = dims
    grid_y, grid_x = np.mgrid[0:h, 0:w]

    kp1, des1 = template_features
    kp2, des2 = image_features

    try:
        selected_kp1 = [kp1[idx1] for idx1 in template_matched_idxs]
        selected_kp2 = [kp2[idx2] for idx2 in image_matched_idxs]

        kp1_arr = np.array([kp.pt for kp in selected_kp1])
        kp2_arr = np.array([kp.pt for kp in selected_kp2])

        x_interp = griddata(kp2_arr, kp1_arr[:, 0], (grid_x, grid_y), method='cubic')
        y_interp = griddata(kp2_arr, kp1_arr[:, 1], (grid_x, grid_y), method='cubic')
    
        flow = np.zeros((h, w, 2))
        flow[:, :, 0] = x_interp
        flow[:, :, 1] = y_interp

    except Exception as e:

        print(e)
        flow = np.nan * np.ones((h, w, 2))

    flow = flow.astype(np.float32)

    return flow


def compute_color_map(dims, image_features, template_features, image_matched_idxs, template_matched_idxs, rot):
    from scipy.interpolate import griddata

    h, w = dims
    grid_y, grid_x = np.mgrid[0:h, 0:w]

    kp1, des1 = template_features
    kp2, des2 = image_features

    if len(template_matched_idxs) == 0 or len(image_matched_idxs) == 0:
        return np.nan * np.ones((h, w, 2))

    selected_kp1 = [kp1[idx1] for idx1 in template_matched_idxs]
    selected_kp2 = [kp2[idx2] for idx2 in image_matched_idxs]

    theta = np.radians(rot)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    selected_kp1_rot = np.array([R.dot(kp.pt) for kp in selected_kp1])
    points = np.array([kp.pt for kp in selected_kp2])

    values_x = selected_kp1_rot[:, 0]
    try:
        x_interp = griddata(points, values_x, (grid_x, grid_y), method='cubic')
    except Exception as e:
        print(e)
        return np.nan * np.ones((h, w, 2))

    values_y = selected_kp1_rot[:, 1]
    try:
        y_interp = griddata(points, values_y, (grid_x, grid_y), method='cubic')
    except Exception as e:
        print(e)
        return np.nan * np.ones((h, w, 2))

    kp1_array = np.array([kp.pt for kp in kp1])
    kp1_rot = np.array([R.dot(kp.pt) for kp in kp1])

    x_max = kp1_rot[:, 0].max()
    x_min = kp1_rot[:, 0].min()
    y_max = kp1_rot[:, 1].max()
    y_min = kp1_rot[:, 1].min()
    
    color_map = np.zeros((h, w, 2))
    color_map[:, :, 0] = 1 - ((x_interp - x_min) / (x_max - x_min))
    color_map[:, :, 1] = (y_interp - y_min) / (y_max - y_min)

    return color_map


def compute_checker_map(image, flow):
    mask = cv2.imread(image.replace('uv', 'mask'))
    nan_mask = np.isnan(flow)
    flow[nan_mask] = 0.0

    checkerboard = checkboard_like(mask, sq=30) * 1.0
    warped_img = cv2.remap(checkerboard, flow, None, cv2.INTER_LINEAR)
    warped_img[mask[:, :, 0] == 0, :] = 0
    warped_img[np.logical_and(mask[:, :, 0] > 0, nan_mask[:, :, 0]), :] = 1

    return warped_img


def compute_color_map_image(image, color_map):
    from skimage.color import hsv2rgb

    mask = cv2.imread(image.replace('uv', 'mask'))
    
    noncolor_mask = np.isnan(color_map[:, :, 0])

    colorized_hsv = np.ones(color_map.shape[0:2] + (3,))
    colorized_hsv[:, :, 0] = color_map[:, :, 0]
    colorized_hsv[:, :, 1] = color_map[:, :, 1]
    colorized_hsv[noncolor_mask] = 0

    colorized = hsv2rgb(colorized_hsv)

    colorized_masked = colorized.copy()
    colorized_masked[mask[:, :, 0] == 0] = 0

    uncolored_hand_area = np.logical_and(mask[:, :, 0] > 0, noncolor_mask)
    colorized_masked[uncolored_hand_area] = 1.0
    
    colorized_masked[np.isnan(colorized_masked)] = 0.0
    colorized_masked = np.clip(colorized_masked, 0.0, 1.0)

    return colorized_masked


def compute_and_save_checker_map(args):
    (image, template_image, output_arr, output_image, rot) = args
    # load image and sift features
    img = load_image(image)
    image_features = sift_list_from_file(image.replace('png', 'key')) 
    # load covering images sift features
    template_features = sift_list_from_file(template_image.replace('png', 'key'))
    # compute matched features
    image_matched_idxs, template_matched_idxs = get_covered_idxs(image_features, template_features)
    # render color map 
    flow = compute_flow(img.shape[0:2], image_features, template_features, image_matched_idxs, template_matched_idxs)
    # save image
    checker_map = compute_checker_map(image, flow)
    print(f'saving output_image: {output_image}')
    plt.imsave(output_image, checker_map)


if __name__ == '__main__':

    # image_list_file = sys.argv[1] 
    # image_list = readlines(image_list_file)

    image_dir = sys.argv[1] 
    template_image = sys.argv[2] 
    output_dir = sys.argv[3]
    rot = int(sys.argv[4])

    image_list = glob.glob(os.path.join(image_dir, '*.png'))

    #with open(covering_file, 'r') as f:
    #    covering_images = [line.strip() for line in f.readlines()]

    args = []
    for image in image_list:
        img_basename = os.path.basename(image)
        arr_basename = os.path.basename(image).replace('png', 'npy')
        output_image = os.path.join(output_dir, img_basename)
        output_arr = os.path.join(output_dir, arr_basename)
        args.append((image, template_image, output_arr, output_image, rot))

    n_cpus = multiprocessing.cpu_count()

    with Pool(n_cpus) as p:
        p.map(compute_and_save_checker_map, args)


