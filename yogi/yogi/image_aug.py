import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import numpy as np


def augment(augmenter, image, joints):
    """Joints are in DeeperCut format.
    K rows, 3 columns: (joint_id, x, y)"""

    if len(joints) > 1:
        raise Exception("only single-person data is currently implemented")

    if len(joints) == 1:
        person_joints = joints[0]
    else:
        assert(len(joints) == 0)
        person_joints = []

    # joints to keypoints
    person_ids = [id for (id, _, _) in person_joints]
    kp_list = [Keypoint(x=x, y=y) for (_, x, y) in person_joints]
    keypoints_on_img = KeypointsOnImage(kp_list, shape=image.shape)

    # augment
    image_aug, keypoints_aug = augmenter(image=image,
                                         keypoints=keypoints_on_img)
    # keypoints to joints
    kp_array = keypoints_aug.to_xy_array()
    person_joints_list = [(id, x, y) for (id, (x, y)) in
                          zip(person_ids, kp_array)]

    # there is probably a more elegant way to do this
    if len(person_joints_list) > 0:
        person_joints_aug = np.array(person_joints_list)
        joints_aug = [person_joints_aug]
    else:
        joints_aug = []

    return image_aug, joints_aug


def make_redify(factor=0.1, prob=0.5):
    seq = iaa.Sequential([
        iaa.Sometimes(prob,
                      iaa.WithChannels([1, 2],
                                       iaa.Multiply(factor)))
    ], random_order=False)
    return seq


def make_noise_redify_gray(warp=False):
    # Example taked from: https://imgaug.readthedocs.io

    aug_list = [
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.ContrastNormalization((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),
                                  per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
    ]

    if warp:
        aug_list.extend([
            iaa.Crop(percent=(0, 0.1)),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-15, 15),
                shear=(-8, 8)
            )
        ])

    seq = iaa.Sequential(aug_list, random_order=True)
    seq = iaa.Sequential([
        seq,
        iaa.Sometimes(0.5,
                      iaa.OneOf([
                          iaa.Grayscale(alpha=1.0),
                          iaa.WithChannels([1, 2], iaa.Multiply(0.1))]))
    ], random_order=False)

    return seq


def make_noise_warp_rotate():
    aug_list = [
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.ContrastNormalization((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),
                                  per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
    ]

    aug_list.extend([
        iaa.Crop(percent=(0, 0.1)),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(0, 360),
            shear=(-8, 8)
        )
    ])

    seq = iaa.Sequential(aug_list, random_order=True)

    return seq


def random_image(shape):

    aug1 = iaa.CoarseSaltAndPepper(0.05, size_px=(4, 32))
    aug2 = make_noise_redify_gray(warp=True)

    image = np.ones(shape, dtype=np.uint8) * 255
    image = iaa.Multiply((0.0, 1.0))(image=image)
    for _ in range(2):
        image = aug1(image=image)
        image = aug2(image=image)
    return image


redify = make_redify()
noise_redify_gray = make_noise_redify_gray(warp=False)
noise_redify_gray_warp = make_noise_redify_gray(warp=True)
noise_warp_rotate = make_noise_warp_rotate()
