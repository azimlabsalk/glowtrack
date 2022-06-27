import os
import sys

cyclegan_path = os.path.join(os.path.dirname(__file__), "3rd-party/pytorch-CycleGAN-and-pix2pix/")
sys.path.append(cyclegan_path + "models")
sys.path.append(cyclegan_path + "util")

from PIL import Image
import torch
from torch.multiprocessing import Pool, Process, set_start_method
from torchvision import transforms

import networks
import util

try:
    set_start_method('spawn')
except RuntimeError:
    pass

class CycleGanNetwork(object):

    def __init__(self):
        self.net = None
        self.create_transform()

    def __call__(self, image):
        input_tensor = self.transform(image)
        input_tensor = input_tensor.unsqueeze(0)
        output_tensor = self.net.forward(input_tensor)
        output_image = util.tensor2im(output_tensor)
        return output_image

    def load(self, path):
        self.net = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance', False,
            'normal', 0.02, [0])
        try:
            state_dict = torch.load(path)
            self.net.module.load_state_dict(state_dict)
        except Exception as inst:
            print(inst)

    def create_transform(self):
        self.transform_list = []

        self.transform_list.append(transforms.ToPILImage())

        self.transform_list.append(transforms.Resize([256,256], Image.BICUBIC))
        # self.transform_list.append(transforms.RandomCrop([256,256]))

        self.transform_list.append(transforms.ToTensor())
        self.transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)))

        self.transform = transforms.Compose(self.transform_list)

    def can_process(self, image):
        result = (self.net is not None)
        return result
