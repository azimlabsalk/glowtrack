import torch
import torch.optim
import math
import numpy as np

import glob
import os
import time

import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import median


def prep_for_sift(img, median_filt=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if median_filt:
        gray = median(gray)
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

from pprint import pprint

templates_file = '/nadata/mnle/data/DAN_BUTLER/dev/yogi/scripts/speckle/covering-images/selected-10.txt'

with open(templates_file, 'r') as f:
    templates = [t.strip() for t in f.readlines()]

pprint(templates)



# load template features
templates_kp = []
templates_des = []
for template in templates:
    keyfile = template.replace('png', 'key')
    (kp, des) = sift_list_from_file(keyfile)
    templates_kp.append(kp)
    templates_des.append(des)



i1 = 0
i2 = 1

file1 = templates[i1]
file2 = templates[i2]

kp1 = templates_kp[i1]
kp2 = templates_kp[i2]

des1 = templates_des[i1]
des2 = templates_des[i2]

img1 = cv2.imread(file1)
img2 = cv2.imread(file2)


def matches_to_edges(neighbor_matches):
    neighbor_edges = []
    for i in range(len(neighbor_matches)):
        for j in range(len(neighbor_matches[i])):
            match = neighbor_matches[i][j]
            edge = (match.trainIdx, match.queryIdx)
            neighbor_edges.append(edge)
    return neighbor_edges


from collections import Counter

def eliminate_multimatches(matches):
    counts1 = Counter([m.queryIdx for m in matches])
    dups1 = set([x for (x, c) in counts1.items() if c > 1])
    counts2 = Counter([m.trainIdx for m in matches])
    dups2 = set([x for (x, c) in counts2.items() if c > 1])
    matches = [m for m in matches if m.queryIdx not in dups1 and m.trainIdx not in dups2]
    return matches

# create match edges
matches = compute_matches((kp1, des1),
                          (kp2, des2),
                          ratio_threshold=0.75,
                          epsilon=50,
                          delta=50)

matches = eliminate_multimatches(matches)

match_idxs1 = np.array([m.queryIdx for m in matches])
match_idxs2 = np.array([m.trainIdx for m in matches])

assert(len(set(match_idxs1)) == len(match_idxs1))
assert(len(set(match_idxs2)) == len(match_idxs2))

# create neighbor graphs

def create_array(keypoints, idxs):
    pts = [keypoints[idx].pt for idx in idxs]
    pos = np.array(pts)
    pos = pos.astype(np.float32)
    return pos

# matching points only
selected_idxs1 = match_idxs1
selected_idxs2 = match_idxs2

# all points
# selected_idxs1 = range(len(kp1))
# selected_idxs2 = range(len(kp2))

pos1_arr = create_array(kp1, selected_idxs1)
pos2_arr = create_array(kp2, selected_idxs2)

bf = cv2.BFMatcher()

k = 10
nn_matches1 = bf.knnMatch(pos1_arr, pos1_arr, k=k)
nn_matches2 = bf.knnMatch(pos2_arr, pos2_arr, k=k)

radius = 50
radius_matches1 = bf.radiusMatch(pos1_arr, pos1_arr, radius)
radius_matches2 = bf.radiusMatch(pos2_arr, pos2_arr, radius)

neighbor_matches1 = [rad_list if len(rad_list) <= k else nn_list
                     for (rad_list, nn_list) in zip(radius_matches1, nn_matches1)]
neighbor_matches2 = [rad_list if len(rad_list) <= k else nn_list
                     for (rad_list, nn_list) in zip(radius_matches2, nn_matches2)]

neighbor_edges1 = matches_to_edges(neighbor_matches1)
neighbor_edges2 = matches_to_edges(neighbor_matches2)

# only keep matches between selected indices
# remap idxs, so new that the max is (len(selected_idxs) - 1)

idx_within_selected1 = dict((idx, i) for (i, idx) in enumerate(selected_idxs1))
idx_within_selected2 = dict((idx, i) for (i, idx) in enumerate(selected_idxs2))

match_idxs = []
for m in matches:
    (idx1, idx2) = (m.queryIdx, m.trainIdx)
    if idx1 in selected_idxs1 and idx2 in selected_idxs2:
        match_idxs.append((idx_within_selected1[idx1],
                           idx_within_selected2[idx2]))


import time

pos1_init = torch.tensor(pos1_arr.T, device=device, dtype=dtype)
pos2_init = torch.tensor(pos2_arr.T, device=device, dtype=dtype)

(d1, n1) = pos1_init.shape[0:2]
(d2, n2) = pos2_init.shape[0:2]

d = 2

assert(d1 == d2 == d == 2)
        
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")

noise1 = 100 * torch.randn((d, n1), device=device, dtype=dtype)
noise2 = 100 * torch.randn((d, n2), device=device, dtype=dtype)

embedding_d = d  # d + 1

pos1 = torch.zeros(torch.Size([embedding_d, n1]), device=device, dtype=dtype)
pos1[:d, :].copy_(pos1_init)
# pos1[:d, :] += noise1
pos1[d:, :] += 0.01 * torch.randn((embedding_d - d, n1), device=device, dtype=dtype)
pos1.requires_grad = True

pos2 = torch.zeros(torch.Size([embedding_d, n2]), device=device, dtype=dtype)
pos2[:d, :].copy_(pos2_init)
# pos2[:d, :] += noise2
pos2[d:, :] += 0.01 * torch.randn((embedding_d - d, n2), device=device, dtype=dtype)
pos2.requires_grad = True

learning_rate = 1
optimizer = torch.optim.LBFGS([pos1, pos2], lr=learning_rate)

import torch.functional as tf

def compute_neighbor_loss(pos, pos_init, neighbor_edges):

    # enforce neighbor distances
    neighbor_loss = torch.tensor((0), device=device, dtype=dtype)
    for (i, j) in neighbor_edges:
        dist = (pos[:, i] - pos[:, j]).pow(2).sum()
        dist_init = (pos_init[:, i] - pos_init[:, j]).pow(2).sum()
        l = (dist - dist_init).pow(2).sum()
        neighbor_loss += l

    return neighbor_loss

def compute_match_loss(pos1, pos2, match_idxs):
    match_loss = torch.tensor((0), device=device, dtype=dtype)
    for (idx1, idx2) in match_idxs:
        l = (pos1[:, idx1] - pos2[:, idx2]).pow(2).sum()
        match_loss += l
    return match_loss

def compute_loss(pos1, pos2, pos1_init, pos2_init, neighbor_edges1, neighbor_edges2, match_idxs, verbose=False):

    # enforce neighbor distances
    neighbor_loss1 = compute_neighbor_loss(pos1, pos1_init, neighbor_edges1)
    neighbor_loss2 = compute_neighbor_loss(pos2, pos2_init, neighbor_edges2)
        
    # enforce match consistency    
    match_loss = compute_match_loss(pos1, pos2, match_idxs)

#     # encourage nonneighbors to be far
#     nonneighbor_loss = torch.tensor((0), device=device, dtype=dtype)    
#     if nonneighbor_term:
#         for (i, j) in nonneighbor_edges:
#             dist = (pos[:, i] - pos[:, j]).pow(2).sum()
#             dist_init = (pos_init[:, i] - pos_init[:, j]).pow(2).sum()
#             l = tf.F.relu(dist_init - dist, 0)
#             nonneighbor_loss += l

#     loss = neighbor_loss + nonneighbor_loss
#     loss = neighbor_loss1 + neighbor_loss2

#     loss = neighbor_loss1 + neighbor_loss2
    loss = neighbor_loss1 + neighbor_loss2 + (10 * match_loss)

    return loss

def closure():
    optimizer.zero_grad()
    loss = compute_loss(pos1, pos2, pos1_init, pos2_init, neighbor_edges1, neighbor_edges2, match_idxs)
    loss.backward()
    return loss

pos1_values = []
pos2_values = []

t0 = time.time()

pos1_values.append(pos1.detach().numpy().copy())
pos2_values.append(pos2.detach().numpy().copy())

for t in range(20):
    
    with torch.no_grad():
        neighbor_loss1 = compute_neighbor_loss(pos1, pos1_init, neighbor_edges1)
        neighbor_loss2 = compute_neighbor_loss(pos2, pos2_init, neighbor_edges2)
        match_loss = compute_match_loss(pos1, pos2, match_idxs)
        loss = neighbor_loss1 + neighbor_loss2 + (10 * match_loss)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
        
    print(f'iteration {t} ({dt:.3f}s) neighbor_loss1: {neighbor_loss1:.3e}, neighbor_loss2: {neighbor_loss2:.3e}, match_loss: {10 * match_loss:.3e}, total: {loss:.3e}')

    optimizer.step(closure)

    pos1_values.append(pos1.detach().numpy())
    pos2_values.append(pos2.detach().numpy())

# for t in range(5):
#     pos_values.append(pos.detach().numpy().copy())
#     loss = compute_loss(pos, pos_init, neighbor_edges, nonneighbor_edges, neighbor_term=True, nonneighbor_term=True)
#     print(f'iteration {t}, loss: {loss}')
#     optimizer.step(closure)


plt.figure(figsize=(20, 10))

plt.subplot(121)
for i in range(n):
    plt.plot([pos1_values[0][0, i], pos2_values[0][0, i]],
             [pos1_values[0][1, i], pos2_values[0][1, i]], color='k', zorder=0.0)
plt.scatter(pos2_values[0][0, :], pos2_values[0][1, :], color='m', zorder=1.0)
plt.scatter(pos1_values[0][0, :], pos1_values[0][1, :], color='c', zorder=1.0)
n = pos2_values[0].shape[1]
plt.axis('equal')
plt.xlim([0, 1440])
plt.ylim([0, 1080])
# plt.axis('off')
plt.title('before warping')

plt.subplot(122)
for i in range(n):
    plt.plot([pos1_values[-1][0, i], pos2_values[-1][0, i]],
             [pos1_values[-1][1, i], pos2_values[-1][1, i]], color='k', zorder=0.0)
plt.scatter(pos2_values[-1][0, :], pos2_values[-1][1, :], color='m', zorder=1.0)
plt.scatter(pos1_values[-1][0, :], pos1_values[-1][1, :], color='c', zorder=1.0)
plt.axis('equal')
plt.xlim([0, 1440])
plt.ylim([0, 1080])
# plt.axis('off')
plt.title('after warping')

