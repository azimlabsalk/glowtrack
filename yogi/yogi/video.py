import os

import imageio
import numpy as np
from skimage.io import imsave


def video_to_frames(video_path, frames_dir, extension='jpg', flip=False):
    reader = imageio.get_reader(video_path)
    frame_paths = []
    for i, image in enumerate(reader.iter_data()):
        fname = '{:08d}.{}'.format(i, extension)
        fpath = os.path.join(frames_dir, fname)
        imsave(fpath, image if not flip else flip_image(image))
        frame_paths.append(fpath)
    reader.close()
    return frame_paths


def flip_image(image):
    return np.fliplr(image)


def get_video_size(video_path):
    reader = imageio.get_reader(video_path)
    (w, h) = reader.get_meta_data()['size']
    reader.close()
    return (w, h)


def render_video(output_path, images, label_source_ids, session,
                 conf_threshold=None, quality=10, fps=30,
                 landmarkset_id=None, color=None, show_conf=True):
    writer = imageio.get_writer(output_path, quality=quality, fps=fps)
    for image in images:
        img = image.render_labels(label_source_ids, session,
                                  conf_threshold=conf_threshold,
                                  landmarkset_id=landmarkset_id,
                                  color=color, show_conf=show_conf)
        writer.append_data(img)
    writer.close()
