import os
import sys

import numpy as np
from scipy.misc import imread, imresize

deepercut_path = os.path.join(os.path.dirname(__file__), "3rd-party/pose-tensorflow/")
sys.path.append(deepercut_path)
#sys.path.append(cyclegan_path + "models")
#sys.path.append(cyclegan_path + "util")

from util.config import load_config
from nnet.predict import setup_pose_prediction, extract_cnn_output, argmax_pose_predict
from dataset.pose_dataset import data_to_input


class DeeperCutNetwork(object):

    def __init__(self):
        self.cfg = None
        self.sess = None
        self.locref = None
        self.pairwise_diff = None

    def __call__(self, image):

        scale = self.cfg.global_scale
        img = imresize(image, scale) if scale != 1 else image

        outputs_np = self.sess.run(self.outputs, feed_dict={self.inputs: data_to_input(img)})
        scmap, locref, pairwise_diff = extract_cnn_output(outputs_np, self.cfg)

        if len(scmap.shape) == 2:
            scmap = np.expand_dims(scmap, axis=2)
        pose = argmax_pose_predict(scmap, locref, self.cfg.stride)
        pose_refscale = np.copy(pose)
        pose_refscale[:, 0:2] /= self.cfg.global_scale
        predictions = pose_refscale

        return scmap, predictions

    def load(self, config_filename):
        try:
            self.cfg = load_config(filename=config_filename)
            self.sess, self.inputs, self.outputs = setup_pose_prediction(self.cfg)
        except Exception as inst:
            print(inst)

    def can_process(self, image):
        result = (self.cfg is not None) and (self.sess is not None)
        return result
