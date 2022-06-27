"""Error Metrics."""
import numpy as np
import matplotlib.pyplot as plt
from os.path import splitext


class LabelError():
    """Error measurement - ground truth v/s predicted labels."""

    def __init__(self, predictions, ground_truth, pixel_rel_size=1. / 848., clip=None):
        """Init error measurement.

        Parameters:
        ----------

        predictions : float
            Predicted labels (x_hat, y_hat, p)

        ground_truth : float
            Ground truth labels (x, y, None)

        pixel_rel_size : float, default = 1/848
            Length of each pixel in terms of prediction label value
            i.e. pixel_rel_size = 1/848 implies
                 848 pixels correspond to predicted label 1.0
        """
        assert ground_truth.shape[0] > 0 and predictions.shape[0] > 0, 'input must contain more than one row.'
        assert ground_truth.shape[1] == 3 and predictions.shape[1] == 3, 'input must contain only 3 columns <x,y,p>.'
        assert len(ground_truth) == len(predictions)
        assert pixel_rel_size > 0

        # Convert to numpy if not already numpy
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        # Ignore obscured labels in ground truth
        obscured_loc = np.equal(ground_truth[:, 0], None)
        non_obscured_loc = np.arange(len(ground_truth))
        non_obscured_loc = non_obscured_loc[np.logical_not(obscured_loc), ...]
        non_obscured_arr = np.expand_dims(non_obscured_loc, axis=1)
        # Store predictions & ground truth as index, x, y, p
        # Ground truth have index only where x,y are not None
        predictions = np.concatenate((np.expand_dims(np.arange(len(ground_truth)), axis=1),
                                      predictions), axis=1)
        ground_truth = np.concatenate((non_obscured_arr,
                                       ground_truth[non_obscured_loc, :]), axis=1)

        self.clip = clip
        self.predictions = predictions
        self.ground_truth = ground_truth
        # Store obscured predictions separately
        obscured_predictions = predictions[obscured_loc, :]
        obscured_predictions[:, 1:-1] = obscured_predictions[:, 1:-1] / pixel_rel_size
        self.obscured_predictions = obscured_predictions

        # Calculate distance related error only for non obscured locations
        # Store error & l2error as index, error, p
        # error = [index, x_hat - x, y_hat - y, confidence] retaining sign
        error = predictions[non_obscured_loc, 1:-1] - ground_truth[:, 1:-1]
        error = np.concatenate((non_obscured_arr,
                                error,
                                predictions[non_obscured_loc, -1:]), axis=1)
        self.error = error

        # euclidean dist = [sqrt((x_hat - x)^2 + (y_hat - y)^2), confidence]
        l2error = np.expand_dims(np.dot(error[:, 1:-1],
                                        np.transpose(error[:, 1:-1])).diagonal(), axis=1) ** 0.5
        l2error = np.concatenate((non_obscured_arr,
                                  l2error,
                                  predictions[non_obscured_loc, -1:]), axis=1)
        self.l2error = l2error
        self.l2error_pixel = l2error.copy()
        self.l2error_pixel[:, 1] = self.l2error_pixel[:, 1] / pixel_rel_size

    def obscured_fp_confidence_histogram(self, bins=None,
                                         plot=True, save_to_file=None):
        """Histogram of the pixel error."""
        if bins is None:
            # 10 logspaced bins
            bins = 'auto'
        if self.clip is None:
            title = 'Histogram of the confidence of fp error'
        else:
            title = 'Histogram of the confidence of fp error for: ' + splitext(self.clip)[0]
        return(histogram(y=self.obscured_predictions[:, -1:],
                         bins=bins,
                         plot=plot,
                         save_to_file=save_to_file,
                         title=title,
                         xlabel='Pixel error bins in logspace'))

    def prediction_error_histogram(self, bins=None,
                                   title_str=None,
                                   plot=True, save_to_file=None):
        """Histogram of the pixel error."""
        if bins is None:
            # 10 logspaced bins
            bins = np.append([0], np.logspace(0, np.log10(max(self.l2error_pixel[:, 1])), num=10))
        if self.clip is None:
            title = 'Histogram of the pixel error'
        else:
            title = 'Histogram of the pixel error for: ' + splitext(self.clip)[0]
        print('prediction_error_histogram ', self.l2error_pixel[0, 1])
        return(histogram(y=self.l2error_pixel[:, 1],
                         bins=bins,
                         plot=plot,
                         save_to_file=save_to_file,
                         title=title,
                         xlabel='Pixel error bins in logspace'))

    def rmse(self):
        """Root mean square pixel error."""
        return(rmse(self.l2error_pixel[:, 1]))

    def PCK(self, pixel_error_threshold=5.,
            plot=True, save_to_file=None):
        """Percentage Correct points wrt pixel error.

        Parameters:
        ----------

        pixel_error_threshold : float, default = 5 pixels
            If euclidean distance between prediction and ground truth > pixel_error_threshold
            Then it is considered a miss

        hist_plot : boolean, default = True
            If True, plot histogram of confidence values of correct points

        Output:
        ----------

        Percentage correct points : Count (l2error <= error_threshold) / Length(l2error)
        mean of confidence value of correct points
        max of confidence value of correct points
        min of confidence value of correct points
        variance of confidence value of correct points
        """
        assert pixel_error_threshold is not None and pixel_error_threshold > 0, 'Incorrect error threshold in pixels. Example: 5 pixels'

        if self.clip is None:
            title = ''
        else:
            title = splitext(self.clip)[0] + ': '

        pck, rmse, stats = PCK(self.l2error_pixel[:, 1:],
                               error_threshold=pixel_error_threshold,
                               plot=plot, save_to_file=save_to_file, title=title)

        return(pck, rmse, stats)


def PCK(error, error_threshold,
        ground_truth_count=None,
        plot=True, save_to_file=None, title=''):
    """Percentage Correct Key-points (PCK) wrt pixel error.

    By default,
        PCK @ error_threshold = count(error <= error_threshold) / length(error)
    Alternately, for PCK with respect to ground truth, provide ground_truth_count
        PCK  = count(error <= error_threshold) / ground_truth_count

    Parameters:
    ----------

    error: float, numpy array
        Error array with dimensions (N, 2) - [[error, confidence]]

    ground_truth_count : float
        Ground truth count without occlusion or out of frame

    error_threshold : float, default = 5 pixels
        If euclidean distance between prediction and ground truth > error_threshold
        Then it is considered a miss

    plot : boolean, default = True
        If True, plot histogram of confidence values of correct points

    save_to_file : str, default = None
        File name to save histogram plots

    title : str, default = ''
        Title for histogram plots

    Output:
    ----------
    tuple
    (PCK @ threshold,
    (mean of confidence value of correct points,
    max of confidence value of correct points,
    min of confidence value of correct points,
    variance of confidence value of correct points))
    """
    assert error_threshold is not None and error_threshold > 0, 'Incorrect error threshold'

    if ground_truth_count is not None:
        count = ground_truth_count
    else:
        count = len(error)

    correct_points = error[error[:, 0] <= error_threshold, :]
    incorrect_points = error[error[:, 0] > error_threshold, :]

    pck = len(correct_points) / count
    rmse_correct = rmse(correct_points[:, 0])
    rmse_incorrect = rmse(incorrect_points[:, 0])

    if (plot is True) or (save_to_file is not None):
        if plot is False:
            # Turn interactive plotting off
            plt.ioff()
        # Create a new figure, plot into it
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(title + 'PCK {0:3.2f} @ Threshold: {1:d}'.format(pck, error_threshold))
        plt.subplot(121)
        bins = np.append([0], np.linspace(0, max(correct_points[:, -1:]), num=10))
        title1 = 'Histogram of correct points - RMSE: {0:3.2f}'.format(rmse_correct)
        xlabel1 = 'Confidence values of correct points'
        _, _ = histogram(correct_points[:, -1:], bins=bins,
                         plot=False)
        plt.title(title1)
        plt.xlabel(xlabel1)
        plt.ylabel('Histogram Count')

        plt.subplot(122)
        bins = np.append([0], np.linspace(0, max(incorrect_points[:, -1:]), num=10))
        title2 = 'Histogram of incorrect points - RMSE: {0:3.2f}'.format(rmse_incorrect)
        xlabel2 = 'Confidence values of incorrect points'
        _, _ = histogram(incorrect_points[:, -1:], bins=bins,
                         plot=False)
        plt.title(title2)
        plt.xlabel(xlabel2)

        if save_to_file is not None:
            plt.savefig(save_to_file)

        if plot is True:
            plt.show()
        else:
            # Close fig so it never gets displayed
            plt.close(fig)

    return(pck,
           (rmse_correct, rmse_incorrect),
           stats(correct_points[:, -1:]))


def rmse(error):
    """Root mean square error.

    Parameters:
    -----------
    error : float numpy
        Error array
    """
    return(np.mean(error ** 2) ** 0.5)


def stats(y):
    """Revert with stats.

    Output:
    ----------

    Percentage correct points : Count (l2error <= error_threshold) / Length(l2error)
    mean of confidence value of correct points
    max of confidence value of correct points
    min of confidence value of correct points
    variance of confidence value of correct points
    """
    return(np.mean(y),
           y.max(),
           y.min(),
           np.var(y))


def histogram(y, bins='auto',
              plot=True,
              save_to_file=None,
              title=None, xlabel=None, ylabel=None):
    """Histogram.

    Parameters:
    ----------

    y : float
        Data of which histogram is to be created

    bins : float array, 'auto', default = 'auto'
        Bins for histogram

    plot : boolean, default = True
        If True, plot histogram

    title, xlabel, ylabel : string
        For Histogram plot
    """
    if title is None:
        title = 'Histogram'

    if xlabel is None:
        xlabel = 'Bins'

    if ylabel is None:
        ylabel = 'Count (Histogram)'

    if (plot is True) or (save_to_file is not None):
        if plot is False:
            # Turn interactive plotting off
            plt.ioff()

        # Create a new figure, plot into it
        fig = plt.figure()
        _, bins, _ = plt.hist(y, bins=bins)
        print('title: ', title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        axes = plt.gca()
        xtick = ["{:.1f}".format(i) for i in bins]
        axes.set_xticks(bins)
        axes.set_xticklabels(xtick)
        plt.xticks(rotation=45)

        if save_to_file is not None:
            plt.savefig(save_to_file)

        if plot is True:
            plt.show()
        else:
            # Close fig so it never gets displayed
            plt.close(fig)
    else:
        _, bins, _ = plt.hist(y, bins=bins)

    return(np.histogram(y, bins=bins))
