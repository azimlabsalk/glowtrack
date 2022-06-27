from scipy import ndimage
from skimage import morphology


def image_to_mask(uv_image, threshold=30, channel=0,
                  erode=False, size_threshold=None):

    mask = uv_image[:, :, channel] > threshold

    if erode:
        mask = morphology.binary_erosion(mask)

    if size_threshold is not None:
        mask = morphology.remove_small_objects(mask,
                                               size_threshold)

    return mask


def get_centroid(mask):
    (h, w) = mask.shape[0:2]
    (y, x) = ndimage.center_of_mass(mask)
    (y, x) = (y / h, x / w)
    return (x, y)


def get_area(mask):
    return mask.sum()


def get_mean_value(uv_image, channel, mask):
    slice = uv_image[:, :, channel]
    return slice[mask].mean()
