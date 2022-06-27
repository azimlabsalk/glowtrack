from yogi.db import session
from yogi.models import Clip


def check_clips():
    for clip in session.query(Clip).all():
        for image in clip.images:
            img = image.get_array()
            if too_dim(img):
                print('has dim frames: {}'.format(str(clip)))
                break


def too_dim(img):
    red_mean = img[:, :, 0].mean()
    is_dim = red_mean < 10
    return is_dim
