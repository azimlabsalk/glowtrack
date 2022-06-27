import numpy as np


def optimize_scale(images, model):

#    scales = [2**p for p in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]]
    scales = [2**p for p in [-1.0, -0.5, 0.0, 0.5, 1.0]]
    print('optimizing over scales: {}'.format(scales))
    (best_scale, _, _) = select_optimal_scale(images, model, scales)

    scales = [best_scale * 2**p for p in [-0.33, -0.16, 0.0, 0.16, 0.33]]
    print('optimizing over scales: {}'.format(scales))
    (best_scale, i, label_arrays) = select_optimal_scale(images, model, scales)

    return best_scale, label_arrays[i]


def optimize_scale_fast(images, model):

#    scales = [2**p for p in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]]
    scales = [2**p for p in [-1.0, -0.5, 0.0, 0.5, 1.0]]
    print('optimizing over scales: {}'.format(scales))
    (best_scale, _, _) = select_optimal_scale_fast(images, model, scales)

    scales = [best_scale * 2**p for p in [-0.33, -0.16, 0.0, 0.16, 0.33]]
    print('optimizing over scales: {}'.format(scales))
    (best_scale, i, label_arrays) = select_optimal_scale_fast(images, model,
                                                              scales)

    return best_scale, label_arrays[i]


def optimize_scale_image(images, model):

    scales = []
    labels = []

    for image in images:

        (scale, label_array) = optimize_scale_fast([image], model)
        scales.append(scale)
        labels.append(label_array[0])

    label_array = np.array(labels)
    return scales, label_array


def select_optimal_scale(images, model, scales):

    label_arrays = []
    speeds = []
    for scale in scales:
        print('computing labels at scale {}'.format(scale))
        label_array = compute_labels(images, model, scale=scale)
        label_arrays.append(label_array)
        speed = mean_speed(label_array)
        speeds.append(speed)

    i_best = np.argmin(speeds)
    best_scale = scales[i_best]

    return (best_scale, i_best, label_arrays)


def select_optimal_scale_fast(images, model, scales):

    images_to_label = images  # [::10]

    label_arrays = []
    confs = []
    for scale in scales:
        print('computing labels at scale {}'.format(scale))
        label_array = compute_labels(images_to_label, model, scale=scale)
        label_arrays.append(label_array)
        conf = mean_conf(label_array)
        confs.append(conf)

    i_best = np.argmax(confs)
    best_scale = scales[i_best]

    return (best_scale, i_best, label_arrays)


def mean_speed(label_array):
    x = label_array[:, 0]
    y = label_array[:, 1]
    delta = np.sqrt(dt(x)**2 + dt(y)**2)
    return delta.mean()


def mean_conf(label_array):
    conf = label_array[:, 2]
    return conf.mean()


def dt(arr):
    return arr[1:] - arr[:-1]


def compute_labels(images, model, scale=None):
    labels = []
    for image in images:
        (scoremap, locref, x, y, confidence) = model.apply(image, prescale=scale)
        labels.append((x, y, confidence))
    labels_arr = np.array(labels)
    return labels_arr

