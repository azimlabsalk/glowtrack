import os

import numpy as np

from yogi.config import pretrained_model_path

default_pretrained = 'resnet_v1_101'
default_iters = 1030000
default_scale = 0.8452830189

default_scale_jitter_lo = 0.85
default_scale_jitter_up = 1.15


class DeeperCut(object):
    """Wraps pose-tensorflow implementation of DeeperCut network."""

    def __init__(self, path):
        self.is_loaded = False
        self.path = path
        self.train_config_file = os.path.join(self.path, 'pose_cfg.yaml')
        self.test_config_file = os.path.join(self.path, 'pose_cfg_test.yaml')

        # These are from pose-tensorflow
        self.cfg = None
        self.sess = None
        self.locref = None
        self.pairwise_diff = None

    def train(self, labelset, landmarkset,
              cuda_visible_devices=None, augmenter=None,
              training_iters=None, global_scale=None, augment_bg=None,
              scale_jitter_lo=None, scale_jitter_up=None, mirror=False):
        from yogi.nn.deepercut_helpers import (YogiPoseDataset,
                                               get_training_schedule,
                                               write_test_config,
                                               write_train_config,
                                               write_labels_file)

        if training_iters is None:
            training_iters = default_iters

        if global_scale is None:
            global_scale = default_scale

        if scale_jitter_lo is None:
            scale_jitter_lo = default_scale_jitter_lo

        if scale_jitter_up is None:
            scale_jitter_up = default_scale_jitter_up

        if augment_bg is None:
            augment_bg = False

        training_schedule = get_training_schedule(training_iters)

        print('Training DeeperCut model ({})'.format(self.path))

        labels_file = os.path.join(self.path, 'training.mat')

        all_joints = landmarkset.all_joints(mirror=mirror)
        print('all_joints = {}'.format(all_joints))

        write_labels_file(labels_file, labelset, landmarkset, mirror)
        write_train_config(self.train_config_file, labels_file, global_scale,
                           scale_jitter_lo, scale_jitter_up,
                           pretrained_model_path, training_schedule,
                           all_joints, mirror)
        checkpoint_path = os.path.join(self.path,
                                       'snapshot-{}'.format(training_iters))
        write_test_config(self.test_config_file, global_scale, checkpoint_path,
                          all_joints)

        # set environment variables
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
        if cuda_visible_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

        try:
            from deepercut.train import train
            from deepercut.util.logging import setup_logging
        except ModuleNotFoundError:
            print('Python module \'deepercut.train\' not found')

            def train():
                print('[Training would happen here]')

        old_path = os.getcwd()
        os.chdir(self.path)

        setup_logging()
        dataset = YogiPoseDataset(labelset=labelset,
                                  augmenter=augmenter,
                                  augment_bg=augment_bg)
        train(dataset=dataset)

        os.chdir(old_path)

    def apply(self, image, return_scoremap=False):
        try:
            from deepercut.nnet.predict import (extract_cnn_output,
                                                argmax_pose_predict)
            from deepercut.dataset.pose_dataset import data_to_input
            from scipy.misc import imresize
        except ModuleNotFoundError:
            print('Python module not found')
            print('Returning empty (x, y, confidence) tuple')
            return (None, None, None)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        scale = self.cfg.global_scale
        img = imresize(image, scale) if scale != 1 else image

        outputs_np = self.sess.run(self.outputs,
                                   feed_dict={self.inputs: data_to_input(img)})
        scmap, locref, pairwise_diff = extract_cnn_output(outputs_np, self.cfg)

        if len(scmap.shape) == 2:
            scmap = np.expand_dims(scmap, axis=2)
        pose = argmax_pose_predict(scmap, locref, self.cfg.stride)
        pose_refscale = np.copy(pose)
        pose_refscale[:, 0:2] /= self.cfg.global_scale
        predictions = pose_refscale

        # assert(predictions.shape[0] == 1)
        confidences = []
        xs = []
        ys = []
        for idx in range(predictions.shape[0]):
            confidence = predictions[idx, 2]
            x, y = predictions[idx, 0:2]

            h, w = image.shape[0:2]
            x = x / w
            y = y / h

            xs.append(x)
            ys.append(y)
            confidences.append(confidence)

        if return_scoremap:
            return (scmap, locref, xs, ys, confidences)
        else:
            return (xs, ys, confidences)

    def load(self):
        try:
            from deepercut.util.config import load_config
            from deepercut.nnet.predict import setup_pose_prediction
        except ModuleNotFoundError:
            print('Python module \'deepercut.train\' not found')
            print('Would load nn here.')
            self.is_loaded = True
            return

        try:
            self.cfg = load_config(filename=self.test_config_file)
            self.sess, self.inputs, self.outputs = setup_pose_prediction(
                self.cfg)
            self.is_loaded = True
        except Exception as inst:
            print(inst)

    def can_process(self, image):
        result = (self.cfg is not None) and (self.sess is not None)
        return result
