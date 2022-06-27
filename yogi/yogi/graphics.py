import matplotlib.cm as cm
import numpy as np
import skimage.draw as skdraw
import cv2


def blend(image, bg_image, bg_mask):
    output_image = image.copy()
    output_image[bg_mask] = bg_image[bg_mask]
    return output_image


def render_labels(img, labels, conf_threshold=None, color=None, show_conf=True):
    img = img.copy()
    radius = img.shape[0] / 100

    color_provided = False

    if color is None:
        colors = cm.get_cmap('Set1').colors[1:]  # skip red
    else:
        color_provided = True
        color = eval(color)
        if type(color) is tuple:
            assert(len(color) == 3)
            colors = [color] * len(labels)
        elif type(color) is list:
            colors = color
        else:
            raise Exception('color should be a list of 3-tuples')

    for i in range(len(labels)):
        label = labels[i]
        if color_provided:
            color = colors[i % len(colors)]
        else:
            landmark_color = labels[i].landmark.color
            if landmark_color is not None:
                color = hex_to_rgb(landmark_color)
            else:
                color = colors[i % len(colors)]

        below_thresh = ((conf_threshold is not None) and
                        (label.confidence is not None) and
                        label.confidence < conf_threshold)
        if not below_thresh and not label.is_hidden():
            render_circle(img, label.x, label.y, radius=radius, color=color)
            if label.confidence is not None and show_conf:
                txt_label = '{:.4f}'.format(label.confidence)
                render_text(img, label.x, label.y, txt_label)

    return img


def render_text(img, x, y, txt_label):
    h, w = img.shape[0:2]
    (x, y) = (int(x*w) + 10, int(y*h) + 10)
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    cv2.putText(img=img, text=txt_label, org=(x, y),
                fontFace=3, fontScale=0.5, color=(0, 0, 255),
                thickness=1)


def hex_to_rgb(s):
    s = [int('0x' + s[i:i+2], base=16) for i in range(1, len(s), 2)]
    return s


def render_circle(img, x, y, radius=2, color=(0, 255, 0)):

    if type(color[0]) is float:
        color = tuple(int(255 * c) for c in color)

    h, w = img.shape[0:2]
    r = int(np.clip(y*h, 0, h))
    c = int(np.clip(x*w, 0, w))
    ii, jj = skdraw.circle(r, c, radius, shape=(h, w))
    img[ii, jj] = color
