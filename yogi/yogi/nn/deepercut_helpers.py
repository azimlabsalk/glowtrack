import logging
import random

import numpy as np
from numpy import array as arr
from numpy import concatenate as cat
import scipy.io as sio
from scipy.io import savemat
from scipy.misc import imread, imresize
from deepercut.dataset.pose_dataset import (collect_pairwise_stats,
                                            data_to_input, Batch, PoseDataset,
                                            DataItem)
from deepercut.util.config import load_config

from yogi.image_aug import augment
from yogi.graphics import blend


def extend_crop(crop, crop_pad, image_size):
    crop[0] = max(crop[0] - crop_pad, 0)
    crop[1] = max(crop[1] - crop_pad, 0)
    crop[2] = min(crop[2] + crop_pad, image_size[2] - 1)
    crop[3] = min(crop[3] + crop_pad, image_size[1] - 1)
    return crop


class YogiPoseDataset(PoseDataset):

    def __init__(self, labelset, config_file=None, augmenter=None,
                 augment_bg=None, augment_bg_strategy=None, augment_bg_prob=0.5):
        if config_file is None:
            cfg = load_config()
        else:
            cfg = load_config(filename=config_file)

        if augment_bg_strategy is None:
            augment_bg_strategy = 'maskfile'
        
        super().__init__(cfg)
        self.augmenter = augmenter
        self.augment_bg = augment_bg
        self.augment_bg_prob = augment_bg_prob
        self.augment_bg_strategy = augment_bg_strategy

    def get_augmented_data(self, data_item):

        im_path = data_item.im_path
        joints = data_item.joints

        image = imread(im_path, mode='RGB')

        if self.augment_bg and self.has_bg_mask(im_path):
            if random.random() < self.augment_bg_prob:
                image = self.get_bg_augmented(im_path, image=image)

        if self.augmenter is not None:
            image, joints = augment(self.augmenter, image, joints)

        return image, joints

    def has_bg_mask(self, im_path):
        return True

    def get_bg_augmented(self, im_path, image=None):
        if image is None:
            image = imread(im_path, mode='RGB')
        bg_mask = self.get_bg_mask(im_path, image=image)
        bg_image = self.random_bg_image(image.shape)
        image_out = blend(image, bg_image, bg_mask)
        return image_out

    def get_bg_mask(self, im_path, image=None):
        from skimage.color import rgb2hsv
        if image is None:
            image = imread(im_path, mode='RGB')
        if self.augment_bg_strategy == 'greenscreen':
            hsv = rgb2hsv(image)
            mask = np.logical_and(hsv[:, :, 0] > 1.5/6, hsv[:, :, 0] < 1.8/6)
            return mask
        elif self.augment_bg_strategy == 'maskfile':
            image = imread(im_path.replace('visible', 'mask'), mode='RGB')
            mask = image[:, :, 0] == 0
            return mask
        else:
            raise Exception("unknown augment_bg_strategy: " + str(augment_bg_strategy))

    def random_bg_image(self, shape):
        from yogi.image_aug import random_image
        img = random_image(shape)
        return img

    def mirror_joints_unknown(self, joints, symmetric_joints):
        res = np.copy(joints)
        joint_id = joints[:, 0].astype(int)
        res[:, 0] = symmetric_joints[joint_id]
        return res

    def load_dataset(self):
        cfg = self.cfg
        file_name = cfg.dataset
        # Load Matlab file dataset annotation
        mlab = sio.loadmat(file_name)
        self.raw_data = mlab
        mlab = mlab['dataset']

        num_images = mlab.shape[1]
        data = []
        has_gt = True

        for i in range(num_images):
            sample = mlab[0, i]

            item = DataItem()
            item.image_id = i
            item.im_path = sample[0][0]
            item.im_size = sample[1][0]
            if len(sample) >= 3:
                joints = sample[2][0][0]
                if joints.shape[0] > 0:
                    joint_id = joints[:, 0]
                    # make sure joint ids are 0-indexed
                    if joint_id.size != 0:
                        assert((joint_id < cfg.num_joints).any())
                    joints[:, 0] = joint_id
                    item.joints = [joints]
                else:
                    item.joints = []
                if len(sample) >= 4:
                    joints_unknown = sample[3][0][0]
                    if joints_unknown.shape[0] > 0:
                        item.joints_unknown = [joints_unknown]
                    else:
                        item.joints_unknown = []
            else:
                has_gt = False
            if cfg.crop:
                crop = sample[3][0] - 1
                item.crop = extend_crop(crop, cfg.crop_pad, item.im_size)
            data.append(item)

        self.has_gt = has_gt
        return data

    def make_batch(self, data_item, scale, mirror):
        assert(self.has_gt)

        im_file = data_item.im_path
        logging.debug('image %s', im_file)
        logging.debug('mirror %r', mirror)

        image, joints = self.get_augmented_data(data_item)

        if self.has_gt:
            joints = np.copy(joints)
            joints_unknown = np.copy(data_item.joints_unknown)

        if self.cfg.crop:
            crop = data_item.crop
            image = image[crop[1]:crop[3] + 1, crop[0]:crop[2] + 1, :]
            if self.has_gt:
                joints[:, 1:3] -= crop[0:2].astype(joints.dtype)

        img = imresize(image, scale) if scale != 1 else image
        scaled_img_size = arr(img.shape[0:2])

        if mirror:
            img = np.fliplr(img)

        batch = {Batch.inputs: img}

        if self.has_gt:
            stride = self.cfg.stride

            if mirror:
                # print('joints = {}'.format(joints))
                joints = [self.mirror_joints(person_joints,
                                             self.symmetric_joints,
                                             image.shape[1])
                          for person_joints in joints]
                # print('joints (mirrored) = {}'.format(joints))

                # print('joints_unknown = {}'.format(joints_unknown))
                joints_unknown = [self.mirror_joints_unknown(
                    person_joints, self.symmetric_joints)
                    for person_joints in joints_unknown]
                data_item.joints_unknown = joints_unknown
                # print('joints_unknown (mirrored) = {}'.format(joints_unknown))

            sm_size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2

            scaled_joints = [person_joints[:, 1:3] * scale for person_joints
                             in joints]

            joint_id = [person_joints[:, 0].astype(int) for person_joints
                        in joints]
            batch = self.compute_targets_and_weights(joint_id, joints_unknown,
                                                     scaled_joints,
                                                     data_item, sm_size, scale,
                                                     batch)

            if self.pairwise_stats_collect:
                data_item.pairwise_stats = collect_pairwise_stats(
                    joint_id, scaled_joints)

        batch = {key: data_to_input(data) for (key, data) in batch.items()}

        batch[Batch.data_item] = data_item

        return batch

    def compute_targets_and_weights(self, joint_id, joints_unknown, coords,
                                    data_item, size, scale, batch):
        stride = self.cfg.stride
        dist_thresh = self.cfg.pos_dist_thresh * scale
        num_joints = self.cfg.num_joints
        half_stride = stride / 2
        scmap = np.zeros(cat([size, arr([num_joints])]))

        locref_shape = cat([size, arr([num_joints * 2])])
        locref_mask = np.zeros(locref_shape)
        locref_map = np.zeros(locref_shape)

        pairwise_shape = cat([size, arr([num_joints * (num_joints - 1) * 2])])
        pairwise_mask = np.zeros(pairwise_shape)
        pairwise_map = np.zeros(pairwise_shape)

        dist_thresh_sq = dist_thresh ** 2

        width = size[1]
        height = size[0]

        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asscalar(joint_pt[0])
                j_y = np.asscalar(joint_pt[1])

                # don't loop over entire heatmap, but just relevant locations
                j_x_sm = round((j_x - half_stride) / stride)
                j_y_sm = round((j_y - half_stride) / stride)
                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))

                for j in range(min_y, max_y + 1):  # range(height):
                    pt_y = j * stride + half_stride
                    for i in range(min_x, max_x + 1):  # range(width):
                        # pt = arr([i*stride+half_stride,
                        #           j*stride+half_stride])
                        # diff = joint_pt - pt
                        # The code above is too slow in python
                        pt_x = i * stride + half_stride
                        dx = j_x - pt_x
                        dy = j_y - pt_y
                        dist = dx ** 2 + dy ** 2
                        # print(la.norm(diff))

                        if dist <= dist_thresh_sq:
                            dist = dx ** 2 + dy ** 2
                            locref_scale = 1.0 / self.cfg.locref_stdev
                            current_normalized_dist = dist * locref_scale ** 2
                            prev_normalized_dist = \
                                locref_map[j, i, j_id * 2 + 0] ** 2 + \
                                locref_map[j, i, j_id * 2 + 1] ** 2
                            update_scores = ((scmap[j, i, j_id] == 0) or
                                             (prev_normalized_dist >
                                              current_normalized_dist))
                            if self.cfg.location_refinement and update_scores:
                                self.set_locref(locref_map, locref_mask,
                                                locref_scale, i, j, j_id,
                                                dx, dy)
                            if self.cfg.pairwise_predict and update_scores:
                                for k_end, j_id_end in enumerate(
                                        joint_id[person_id]):
                                    if k != k_end:
                                        self.set_pairwise_map(
                                            pairwise_map, pairwise_mask, i, j,
                                            j_id, j_id_end, coords, pt_x, pt_y,
                                            person_id, k_end)
                            scmap[j, i, j_id] = 1

        scmap_weights = self.compute_scmap_weights(scmap.shape, joint_id,
                                                   joints_unknown, data_item)

        # Update batch
        batch.update({
            Batch.part_score_targets: scmap,
            Batch.part_score_weights: scmap_weights
        })
        if self.cfg.location_refinement:
            batch.update({
                Batch.locref_targets: locref_map,
                Batch.locref_mask: locref_mask
            })
        if self.cfg.pairwise_predict:
            batch.update({
                Batch.pairwise_targets: pairwise_map,
                Batch.pairwise_mask: pairwise_mask
            })

        return batch

    def compute_scmap_weights(self, scmap_shape, joint_id, joints_unknown,
                              data_item):
        cfg = self.cfg
        if cfg.weigh_only_present_joints:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
            # print('weight.sum() = {}'.format(weights.sum()))
            for person_joint_id in joints_unknown:
                for j_id in person_joint_id:
                    # print('zeroing joint {}'.format(j_id))
                    weights[:, :, j_id] = 0.0
            # print('weight.sum() (after zeroing unknowns) = {}'.format(
            #     weights.sum()))
        return weights


def get_training_schedule(training_iters):
    default_schedule = [
        [0.005, 10000],
        [0.02, 430000],
        [0.002, 730000],
        [0.001, 1030000]
    ]
    default_iters = default_schedule[-1][1]
    scale_factor = training_iters / default_iters
    for i in range(len(default_schedule)):
        default_schedule[i][1] = int(default_schedule[i][1] * scale_factor)
    return default_schedule


def write_labels_file(labels_file, labelset, landmarkset, mirror):
    from collections import Counter
    from sqlalchemy.orm import object_session
    from yogi.models import (Image, Label, LabelSet, Landmark, LandmarkSet,
                             labelset_association_table,
                             landmarkset_association_table)
    from yogi.utils import contains_duplicates

    # print('loading labelset landmark ids')
    # labelset_landmark_ids = set([label.landmark_id for label
    #                              in labelset.labels])

    # print('loading landmarkset landmark ids')
    # landmarkset_landmark_ids = set([landmark.id for landmark
    #                                 in landmarkset.landmarks])

    # assert(labelset_landmark_ids.issubset(landmarkset_landmark_ids))

    print('loading labeled images')
    session = object_session(labelset)
    labeled_images = session.query(Image, Label)\
                            .filter(Label.image_id == Image.id)\
                            .join(labelset_association_table)\
                            .join(LabelSet)\
                            .filter(LabelSet.id == labelset.id)\
                            .filter(Label.landmark_id == Landmark.id)\
                            .all()

    # sanity check
    image_paths = [image.path for (image, label) in labeled_images]
    if contains_duplicates(image_paths):
        print('dataset contains multiple labels per image')

    # group labels by image
    print('grouping labels by image')
    images = set([image for (image, _) in labeled_images])
    image_labels = {} 
    for image in images:
        image_labels[image.id] = []
    for (image, label) in labeled_images:
        image_labels[image.id].append(label)

    # construct deepercut matfile object
    if mirror:
        assert(landmarkset.is_mirrorable())

    landmark_id_list = landmarkset.ids(mirror=mirror)

    print('creating deepercut matfile')
    data_items = []
    n_landmarks = len(landmarkset.landmarks)
    for (i, image) in enumerate(images):
        if i % 100 == 0:
            print(i)
        joints_list = []
        j_ids = []
        for label in image_labels[image.id]:
            (w, h) = (image.width, image.height)
            (x, y) = (label.x, label.y)
#            j_id = landmarks.index(label.landmark_id, mirror)
            j_id = landmark_id_list.index(label.landmark_id)
            j_ids.append(j_id)
            # note: 'None' indicates occlusion
            if x is not None:
                joints_list.append((j_id, int(x * w), int(y * h)))
        # joints_unknown will be ignored by the objective function
        joints_unknown = [i for i in range(n_landmarks) if i not in j_ids]
        data_item = create_data_item(image.path, w, h, joints_list,
                                     joints_unknown)
        data_items.append(data_item)

    dataset_arr = np.array([data_items], dtype=[('image', 'O'), ('size', 'O'),
                                                ('joints', 'O'),
                                                ('joints_unknown', 'O')])
    mdict = {'dataset': dataset_arr}
    savemat(labels_file, mdict)


def create_data_item(fname, w, h, joints_list, joints_unknown):
    # format fname
    img_fname = np.array([fname], dtype='<U')

    size = np.array([[3, h, w]], dtype=np.uint16)

    joints_arr = np.array(joints_list, dtype=np.int32)
    joints = np.array([[0]], dtype=np.object)
    joints[0, 0] = joints_arr

    joints_unknown_arr = np.array(joints_unknown, dtype=np.int32)
    joints_unknown = np.array([[0]], dtype=np.object)
    joints_unknown[0, 0] = joints_unknown_arr

    return (img_fname, size, joints, joints_unknown)


pose_cfg_template = """dataset: {}
num_joints: {}
all_joints: {}

pos_dist_thresh: 17
global_scale: {}
scale_jitter_lo: {}
scale_jitter_up: {}

net_type: resnet_101
init_weights: {}

location_refinement: true
locref_huber_loss: true
locref_loss_weight: 0.05
locref_stdev: 7.2801

intermediate_supervision: true
intermediate_supervision_layer: 12

weigh_part_predictions: true

max_input_size: 850
mirror: {}
multi_step:
- {}
- {}
- {}
- {}
display_iters: 20
save_iters: 10000
# 60000
"""
# weigh_part_predictions: true


def write_train_config(config_file, labels_file, global_scale,
                       scale_jitter_lo, scale_jitter_up,
                       pretrained_model_path, training_schedule, all_joints,
                       mirror):
    num_joints = sum([len(joint_group) for joint_group in all_joints])
    content = pose_cfg_template.format(labels_file, num_joints, all_joints,
                                       global_scale,
                                       scale_jitter_lo, scale_jitter_up,
                                       pretrained_model_path,
                                       mirror,
                                       *training_schedule)
    with open(config_file, 'w') as f:
        f.write(content)


test_cfg_template = """global_scale: {}
init_weights: {}
num_joints: {}
all_joints: {}
location_refinement: true
locref_stdev: 7.2801
net_type: resnet_101
scoremap_dir: clip_0_scoremaps
"""


def write_test_config(config_file, global_scale, checkpoint_path, all_joints):
    num_joints = sum([len(joint_group) for joint_group in all_joints])
    content = test_cfg_template.format(global_scale, checkpoint_path,
                                       num_joints, all_joints)
    with open(config_file, 'w') as f:
        f.write(content)
