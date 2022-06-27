"""Post Prediction Processing Utilities."""
import numpy as np
import math
import pandas as pd
import statsmodels.api as sm


def temporal_correction(input,
                        type='adjacent_mean',
                        **kwargs):
    """
    Apply temporal correction.

    Parameters
    ----------

    input : float (n x 3)
        Dimensionality of the input for <x, y, confidence>

    type : string, default = 'adjacent_mean'
        Type temporal correction algorithm to be used
        choices=['mean', 'ewma', 'lowess']
        Coming soon ['kalman_filter', 'eigen_intersection']
    """
    assert input.shape[0] > 0, 'input must contain more than one row.'
    assert input.shape[1] == 3, 'input must contain only 3 columns <x,y,p>.'

    input = np.array(input)

    if type == 'mean':
        output = adjacent_mean_smoother(input, **kwargs)
    elif type == 'ewma':
        output = exponentially_weighted_average(input, **kwargs)
    elif type == 'lowess':
        output = lowess(input, **kwargs)

    return output


def median_filter(arr, kernel_size=7):
    from scipy.signal import medfilt
    arr = arr.copy()
    arr[:, 0] = medfilt(arr[:, 0], kernel_size=kernel_size)
    arr[:, 1] = medfilt(arr[:, 1], kernel_size=kernel_size)
    return arr


def adjacent_mean_smoother(input, tp_threshold):
    """
    Adjacent_mean_smoother.

    Returns the average of the previous and the next point
    for correcting current predictions if p < tp_threshold. It ignores the
    prediction of the current point.

    Assumption: adjacent_mean_smoother assumes that the adjacent points have
    accurate prediction.

    Caution: Use only when no more than 1 contiguous points have incorrect
    prediction
    """
    if tp_threshold is None:
        tp_threshold = 0.0  # Does nothing. Ignores all points

    # Shift the matrix by 1 row down
    input1 = np.concatenate((np.expand_dims(input[0, :], axis=0),
                             input[:-1, :]), axis=0)
    # Shift the matrix by 1 row up
    input2 = np.concatenate((input[1:, :], input[-1:, :]), axis=0)
    # Calculate mean
    # TODO Compare with x_mean = (x1 + x2) / 2.0
    input_mean = np.mean(np.array([input1, input2]), axis=0)

    output = np.where(input[:, -1:] >= tp_threshold, input, input_mean)
    return output


def exponentially_weighted_average(input,
                                   alpha=None, window=None):
    """
    Exponentially weighted average smoother.

    Use this smoother when the predictions are above threshold but noisy,
    such as when the predictions alternate around the trajectory

    Given input x = [x0, x1, ... xt] and scaling factor and alpha,
    exponentially weighted average is calculated as follows:
    v0 = initial value
    v1 = alpha * v0 + (1 - alpha) * x1
    ...
    vt = alpha * vt-1 + (1 - alpha) * xt

    Scaling factor, alpha and moving average window are related to each other:
    alpha ** window = 1/e,
    i.e.
    alpha = (1 / e) ** (1 / window)

    Either alpha or window must be provided.

    Parameters:
    input : numpy float (n x 3)
        Dimensionality of the input for <x, y, confidence>

    alpha : float
        Moving average scaling factor

    window : float
        Moving average window
    """
    if alpha is None:
        assert window is not None, 'Either alpha or window should be provided'
        assert window > 0, 'Window should be greater than 0'
        alpha = (1 / math.exp(1)) ** (1 / window)

    assert alpha > 0 and alpha <= 1, 'Alpha should be > 0 and <= 1'
    df = pd.DataFrame(input)
    output = np.array(df.ewm(alpha=alpha).mean())
    return(output)


def lowess(input,
           frac=None,
           num_adjacent=None,
           it=3):
    """
    Lowess (or Loess) smoothing.

    LOESS or LOWESS is non-parametric regression method that combine multiple
    regression models in a k-nearest-neighbor-based estimate.

    LOESS combines much of the simplicity of linear least squares regression
    with the flexibility of nonlinear regression. It does this by fitting
    simple models to localized subsets of the data to build up a function that
    describes the variation in the data, point by point.

    Using implementation from https://www.statsmodels.org
    statsmodels.nonparametric.smoothers_lowess.lowess(endog, exog,
                                                      frac=2./3., it=3,
                                                      delta=0.0,
                                                      is_sorted=False,
                                                      missing='drop',
                                                      return_sorted=True)

    TODO:
    Further improvement can be made by ignoring the predictions with confidence
    score < false positive threshold and using the predictions with
    confidence score >= true positive threshold as-is
    """
    lowess = sm.nonparametric.lowess

    if frac is None:
        assert num_adjacent is not None, 'Either frac or num_adjacent ' +\
                                         'should be provided'
        assert num_adjacent > 0, 'num_adjacent should be greater than 0'
        frac = num_adjacent / len(input)

    assert frac > 0 and frac < 1, 'frac should be > 0 and < 1'

    x = input[:, 0]
    y = input[:, 1]

    y_smooth = lowess(y, x, frac=frac, it=it, return_sorted=False)
    output = np.copy(input)
    output[:, 2] = y_smooth
    return(output)
