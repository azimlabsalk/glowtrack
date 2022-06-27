#!/usr/bin/env python3
from yogi.db import session
from yogi.models import ClipSet

clipset_names = [
    'cerebro-paired',
    'cerebro-paired-nondominant',
    'cerebro-paired-handheld',
    'cerebro-paired-rope',
    'cerebro-paired-free',
]


def is_red(image):
    img = image.get_array()
    r_val = img[:, :, 0].mean()
    b_val = img[:, :, 2].mean()
    redness = r_val / b_val
    return redness > 10


def bool_list_to_str(lst):
    return ''.join([str(int(l)) for l in lst])


def find(lst, sublst):
    lst_str = bool_list_to_str(lst)
    sublst_str = bool_list_to_str(sublst)
    index = lst_str.index(sublst_str)
    return index


for name in clipset_names:
    clipset_name = name + '-color'

    try:
        clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    except Exception:
        print('could not find ClipSet ' + clipset_name)
        continue

    print('processing ' + clipset_name)

    full_seq = [1, 1, 1, 1, 0, 0, 0, 0, 0]
    seq_len = len(full_seq)

    for clip in clipset.clips:
        print(clip.path)

        image_is_red = []
        for image in clip.images:
            image_is_red.append(is_red(image))

        offset = find(image_is_red, full_seq)

        n_images = len(image_is_red)
        light_indexes = [(x - offset) % seq_len for x in range(n_images)]

        for (light_index, image) in zip(light_indexes, clip.images):
            image.light_index = light_index

        session.commit()
