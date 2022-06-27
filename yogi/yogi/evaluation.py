import random

from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics

# from yogi.db import session
from yogi.models import LandmarkSet, LabelSet
from yogi.sql import get_labels


def roc_comparison(imageset_name, source_names_pred, source_name_gt,
                   landmarkset_name, output_path, log_fp=False, label=None,
                   epsilon=0.000001, error_threshold=0.05,
                   subsample_factor=10):

    fig = plt.figure(figsize=(10, 10))

    for source_name_pred in source_names_pred:
        print(source_name_pred)
        plot_roc(imageset_name=imageset_name,
                 source_name_pred=source_name_pred,
                 source_name_gt=source_name_gt,
                 landmarkset_name=landmarkset_name,
                 label=source_name_pred,
                 error_threshold=error_threshold,
                 log_fp=log_fp,
                 epsilon=epsilon,
                 subsample_factor=subsample_factor,)

    plt.grid(which='both')
    plt.yticks(np.arange(0, 1.1, step=0.1))

    plt.savefig(output_path)
    plt.close(fig)


def pr_comparison(imageset_name, source_names_pred, source_name_gt,
                  landmarkset_name, output_path, label=None,
                  epsilon=0.000001, error_threshold=0.05,
                  subsample_factor=10):

    fig = plt.figure(figsize=(10, 10))

    for source_name_pred in source_names_pred:
        print(source_name_pred)
        plot_pr(imageset_name=imageset_name,
                source_name_pred=source_name_pred,
                source_name_gt=source_name_gt,
                landmarkset_name=landmarkset_name,
                label=source_name_pred,
                error_threshold=error_threshold,
                epsilon=epsilon,
                subsample_factor=subsample_factor,)

    plt.grid(which='both')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 1.1, step=0.1))

    plt.title('dataset: {}, landmarks: {}'.format(imageset_name, landmarkset_name))

    plt.savefig(output_path)
    plt.close(fig)


def pr_comparison_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, label=None,
                  epsilon=0.000001, error_threshold=0.05,
                  subsample_factor=10, no_conf=False):

    fig = plt.figure(figsize=(10, 10))

    for source_name_pred in source_names_pred:
        print(source_name_pred)
        plot_pr_labelset(imageset_name=imageset_name,
                source_name_pred=source_name_pred,
                gt_labelset_name=gt_labelset_name,
                landmarkset_name=landmarkset_name,
                label=source_name_pred,
                error_threshold=error_threshold,
                epsilon=epsilon,
                subsample_factor=subsample_factor,
                no_conf=no_conf)

    plt.grid(which='both')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 1.1, step=0.1))

    plt.title('dataset: {}, landmarks: {}'.format(imageset_name, landmarkset_name))

    plt.savefig(output_path)
    plt.close(fig)


def auc_comparison_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, label=None,
                  epsilon=0.000001, error_threshold=0.05,
                  subsample_factor=10):

    fig = plt.figure(figsize=(10, 10))

    auc_values = []
    for source_name_pred in source_names_pred:
        print(source_name_pred)
        auc = plot_pr_labelset(imageset_name=imageset_name,
                source_name_pred=source_name_pred,
                gt_labelset_name=gt_labelset_name,
                landmarkset_name=landmarkset_name,
                label=source_name_pred,
                error_threshold=error_threshold,
                epsilon=epsilon,
                subsample_factor=subsample_factor,)
        auc_values.append(auc)

    plt.close('all')

    fig = plt.figure(figsize=(10, 10))

    plt.bar(x=source_names_pred,
            height=auc_values,
            #yerr=[[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.1]],
            capsize=5.0,
            width=0.8)

    plt.xticks([])    

    plt.title('AUC (dataset: {}, landmarks: {})'.format(imageset_name, landmarkset_name))

    plt.savefig(output_path)
    plt.close(fig)


def error_histogram_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, label=None,
                  epsilon=0.000001, error_threshold=0.05, confidence_thresh=0):

    from yogi.db import session

    fig = plt.figure(figsize=(10, 10))

    # assert(len(source_names_pred) == 1)
    for source_name_pred in source_names_pred:

        landmarkset = session.query(LandmarkSet).filter_by(
            name=landmarkset_name).one()

        ground_truth = []
        predictions = []

        for landmark in landmarkset.landmarks:

            landmark_id = landmark.id

            gt = get_labels(labelset_name=gt_labelset_name,
                            imageset_name=imageset_name,
                            landmark_id=landmark_id)

            pred = get_labels(source_name=source_name_pred,
                              imageset_name=imageset_name,
                              landmark_id=landmark_id)

            gt_ids = [label.image_id for label in gt]
            pred_ids = [label.image_id for label in pred]

            common_ids = set(gt_ids).intersection(set(pred_ids))

            gt_dict = dict((label.image_id, label) for label in gt)
            pred_dict = dict((label.image_id, label) for label in pred)

            gt = [gt_dict[image_id] for image_id in common_ids]
            pred = [pred_dict[image_id] for image_id in common_ids]

            ground_truth.extend(gt)
            predictions.extend(pred)

        # pred = [(label.x, label.y, label.confidence) for label in predictions]
        # truth = [(label.x, label.y) for label in ground_truth]

        print('# predictions = {}'.format(len(predictions)))
        print('# ground truth = {}'.format(len(ground_truth)))

        x = []
        y = []
        for (label_p, label_t) in zip(predictions, ground_truth):
            if label_p.confidence >= confidence_thresh and not label_t.is_hidden():
                x.append(label_p.x - label_t.x)
                y.append(label_p.y - label_t.y)


        # PLOT HISTOGRAM HERE
        import seaborn as sns
        #sns.jointplot(x=x, y=y, xlim=(-0.05, 0.05), ylim=(-0.05, 0.05), kind="hist", color="#4CB391")
        d = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)
        sns.histplot(d, bins=np.arange(0, 0.1, 0.002), element="step", fill=False, label=source_name_pred)

    plt.title('pixel error' + (" (confidence threshold = {})".format(confidence_thresh)))
    plt.legend()
    plt.savefig(output_path)
    plt.close(fig)


def compute_error_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, label=None,
                  epsilon=0.000001, error_threshold=0.05, confidence_thresh=0, **kwargs):

    from yogi.db import session

    # fig = plt.figure(figsize=(3, 5))

    data = []
    conf = []

    # assert(len(source_names_pred) == 1)
    for source_name_pred in source_names_pred:

        landmarkset = session.query(LandmarkSet).filter_by(
            name=landmarkset_name).one()

        ground_truth = []
        predictions = []

        for landmark in landmarkset.landmarks:

            landmark_id = landmark.id

            gt = get_labels(labelset_name=gt_labelset_name,
                            imageset_name=imageset_name,
                            landmark_id=landmark_id)

            pred = get_labels(source_name=source_name_pred,
                              imageset_name=imageset_name,
                              landmark_id=landmark_id)

            gt_ids = [label.image_id for label in gt]
            pred_ids = [label.image_id for label in pred]

            common_ids = set(gt_ids).intersection(set(pred_ids))

            gt_dict = dict((label.image_id, label) for label in gt)
            pred_dict = dict((label.image_id, label) for label in pred)

            gt = [gt_dict[image_id] for image_id in common_ids]
            pred = [pred_dict[image_id] for image_id in common_ids]

            ground_truth.extend(gt)
            predictions.extend(pred)

        # pred = [(label.x, label.y, label.confidence) for label in predictions]
        # truth = [(label.x, label.y) for label in ground_truth]

        print('# predictions = {}'.format(len(predictions)))
        print('# ground truth = {}'.format(len(ground_truth)))

        x = []
        y = []
        c = []
        for (label_p, label_t) in zip(predictions, ground_truth):
            if ((confidence_thresh == 0) or label_p.confidence >= confidence_thresh) and not label_t.is_hidden() and not label_p.is_hidden():
    #             x.append(label_p.x - label_t.x)
    #             y.append(label_p.y - label_t.y)
                x.append(label_p.x_px - label_t.x_px)
                y.append(label_p.y_px - label_t.y_px)
                c.append(label_p.confidence)

        print('# of comparisons = {}'.format(len(x)))

        # PLOT HISTOGRAM HERE
        import seaborn as sns

        d = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)
        data.append(d)
        conf.append(c)

    return (data, conf)


def error_boxplot_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs):
    import seaborn as sns
    import numpy as np

    if 'confidence_thresh' not in kwargs:
        kwargs['confidence_thresh'] = 0.0

    logplot = kwargs.get('logplot', True)
    ymax = kwargs.get('ymax', 25)
    whis = kwargs.get('whis', True)
    whis = np.inf if whis else None


    (data, conf) = compute_error_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs)

    # labels = ['original', 'interpolated']
    labels = source_names_pred

    # plt.figure(figsize=(3, 5))
    #result = plt.boxplot(x=data, labels=labels, sym='k.', notch=False, vert=True, whis=(0, 100))
    sns.boxplot(data=data, whis=whis)
    # plt.legend(result['medians'], labels)
    # plt.legend()

    if logplot:
        plt.yscale('log')
        plt.ylim([0.05, 2000])
    else:
        plt.ylim([0, ymax])

    plt.grid(True, axis='y')

    plt.ylabel('error (pixels)')
    plt.title('Pixel error (conf thresh = {})'.format(kwargs['confidence_thresh']))

    # plt.xticks([])
    print(str(len(data)))
    medians = ['{:.2f}'.format(np.median(row)) for row in data]
    locs, _ = plt.xticks()
    plt.xticks(ticks=locs, labels=medians)

    plt.tight_layout()
    # plt.title('pixel error' + (" (confidence threshold = {})".format(confidence_thresh)))

    plt.savefig(output_path)

def error_boxplot_labelset_data(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs):
    import seaborn as sns
    import numpy as np

    if 'confidence_thresh' not in kwargs:
        kwargs['confidence_thresh'] = 0.0

    logplot = kwargs.get('logplot', True)
    ymax = kwargs.get('ymax', 25)
    whis = kwargs.get('whis', True)
    whis = np.inf if whis else None


    (data, conf) = compute_error_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs)

    medians = ['{:.4f}'.format(m) for m in np.median(data, axis=1)]
    return medians

def error_violinplot_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs):
    import seaborn as sns
    import numpy as np

    if 'confidence_thresh' not in kwargs:
        kwargs['confidence_thresh'] = 0.0


    (data, conf) = compute_error_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs)

    # labels = ['original', 'interpolated']
    labels = source_names_pred

    # plt.figure(figsize=(3, 5))
    result = sns.violinplot(data=data, whis=np.inf)

    #plt.yscale('log')
    #plt.ylim([0.05, 2000])
    plt.ylim([0.0, 100.0])
    plt.grid(True, axis='y')

    plt.ylabel('error (pixels)')
    plt.title('Pixel error (conf thresh = {})'.format(kwargs['confidence_thresh']))

    plt.xticks([])

    plt.tight_layout()
    # plt.title('pixel error' + (" (confidence threshold = {})".format(confidence_thresh)))

    plt.savefig(output_path)


def error_cdf_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs):
    import seaborn as sns

    if 'confidence_thresh' not in kwargs:
        kwargs['confidence_thresh'] = 0.0

    # fig = plt.figure(figsize=(3, 5))

    (data, conf) = compute_error_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs)

    sns.ecdfplot(dict(zip(source_names_pred, data)))
    plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.xlim(0, 50)
    title='Error CDF (dataset = "{}", conf thresh = {})'.format(gt_labelset_name, kwargs['confidence_thresh'])
    #if len(title) > 40:
    #    title = title[0:40] + '\n' + title[40:]
    plt.title(title, wrap=True)
    plt.xlabel('Error (pixels)')
    plt.grid(True)

    plt.savefig(output_path)


def error_pvalues_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs):
    from scipy import stats

    if 'confidence_thresh' not in kwargs:
        kwargs['confidence_thresh'] = 0.0

    (data, conf) = compute_error_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, **kwargs)

    n_sources = len(source_names_pred)
    for i in range(n_sources):
        for j in range(n_sources):
            if i < j:
                result = stats.ks_2samp(data[i], data[j])
                print('p_KS({}, {}) = {}'.format(source_names_pred[i], source_names_pred[j], result.pvalue))
                result = stats.anderson_ksamp([data[i], data[j]])
                print('p_AD({}, {}) = {}'.format(source_names_pred[i], source_names_pred[j], result.significance_level))


def plot_roc(imageset_name, source_name_pred, source_name_gt, landmarkset_name,
             log_fp=False, label=None, epsilon=0.000001, error_threshold=0.01,
             subsample_factor=10):

    from yogi.db import session

    landmarkset = session.query(LandmarkSet).filter_by(
        name=landmarkset_name).one()

    ground_truth = []
    predictions = []
    for landmark in landmarkset.landmarks:
        landmark_id = landmark.id
        gt = get_labels(source_name=source_name_gt,
                        imageset_name=imageset_name,
                        landmark_id=landmark_id)
        gt = list(sorted(gt, key=lambda l: l.image_id))

        pred = get_labels(source_name=source_name_pred,
                          imageset_name=imageset_name,
                          landmark_id=landmark_id)
        pred = list(sorted(pred, key=lambda l: l.image_id))

        ground_truth.extend(gt)
        predictions.extend(pred)

    conf_values = [label.confidence for label in predictions]
    conf_thresholds = np.unique(conf_values)
    conf_thresholds = conf_thresholds[::subsample_factor]

    pred = [(label.x, label.y, label.confidence) for label in predictions]
    truth = [(label.x, label.y) for label in ground_truth]

    fp_rates = []
    detection_rates = []
    for conf_threshold in conf_thresholds:
        fpr = false_positive_rate(pred, truth, conf_threshold)
        dr = detection_rate(pred, truth, conf_threshold, error_threshold)
        fp_rates.append(fpr)
        detection_rates.append(dr)

    fp_rates = np.array(fp_rates)
    detection_rates = np.array(detection_rates)

    if log_fp:
        fpr = np.log10(fpr + epsilon)

    plt.plot(fp_rates, detection_rates, label=label)

    plt.xlabel('false positive rate')
    plt.ylabel('detection rate')
    title_template = ('detection rate (error < {} x image width)\n'
                      'vs. false positive rate')
    plt.title(title_template.format(error_threshold))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()


def false_positive_rate(predictions, ground_truth, confidence_threshold):
    assert(len(predictions) == len(ground_truth))

    num_hidden = 0
    num_fp = 0

    for i in range(len(predictions)):
        (x_pred, y_pred, conf) = predictions[i]
        (x, y) = ground_truth[i]
        hidden = (x is None)
        if hidden:
            num_hidden += 1
            if conf > confidence_threshold:
                num_fp += 1

    return num_fp / num_hidden


def detection_rate(predictions, ground_truth, confidence_threshold,
                   error_threshold):
    assert(len(predictions) == len(ground_truth))

    num_present = 0
    num_ontarget = 0

    for i in range(len(predictions)):
        (x_pred, y_pred, conf) = predictions[i]
        (x, y) = ground_truth[i]
        hidden = (x is None)
        if not hidden:
            num_present += 1
            if x_pred is not None and (confidence_threshold is None or conf > confidence_threshold): 
                err = np.sqrt((x_pred - x) ** 2 + (y_pred - y) ** 2)
                if err < error_threshold:
                    num_ontarget += 1

    return num_ontarget / num_present


recall = detection_rate


def precision(predictions, ground_truth, confidence_threshold,
              error_threshold):

    assert(len(predictions) == len(ground_truth))

    num_predictions = 0
    num_ontarget = 0

    for i in range(len(predictions)):
        (x_pred, y_pred, conf) = predictions[i]
        (x, y) = ground_truth[i]
        hidden = (x is None)
        if x_pred is not None and (confidence_threshold is None or conf > confidence_threshold):
            num_predictions += 1
            if not hidden:
                err = np.sqrt((x_pred - x) ** 2 + (y_pred - y) ** 2)
                if err < error_threshold:
                    num_ontarget += 1

    return num_ontarget / num_predictions if num_predictions > 0 else np.nan


def plot_pr(imageset_name, source_name_pred, source_name_gt, landmarkset_name,
            label=None, epsilon=0.000001, error_threshold=0.01,
            subsample_factor=10):
 
    from yogi.db import session

    landmarkset = session.query(LandmarkSet).filter_by(
        name=landmarkset_name).one()

    ground_truth = []
    predictions = []

    for landmark in landmarkset.landmarks:

        landmark_id = landmark.id

        gt = get_labels(source_name=source_name_gt,
                        imageset_name=imageset_name,
                        landmark_id=landmark_id)

        pred = get_labels(source_name=source_name_pred,
                          imageset_name=imageset_name,
                          landmark_id=landmark_id)

        gt_ids = [label.image_id for label in gt]
        pred_ids = [label.image_id for label in pred]

        common_ids = set(gt_ids).intersection(set(pred_ids))

        gt_dict = dict((label.image_id, label) for label in gt)
        pred_dict = dict((label.image_id, label) for label in pred)

        gt = [gt_dict[image_id] for image_id in common_ids]
        pred = [pred_dict[image_id] for image_id in common_ids]

        ground_truth.extend(gt)
        predictions.extend(pred)

    conf_values = [label.confidence for label in predictions]
    conf_thresholds = np.unique(conf_values)[:-1]
    conf_thresholds = conf_thresholds[::subsample_factor]

    pred = [(label.x, label.y, label.confidence) for label in predictions]
    truth = [(label.x, label.y) for label in ground_truth]

    print('number of predictions = {}'.format(len(pred)))
    print('number of labels = {}'.format(len(truth)))

    precisions = []
    recalls = []
    for conf_threshold in conf_thresholds:
        p = precision(pred, truth, conf_threshold, error_threshold)
        r = recall(pred, truth, conf_threshold, error_threshold)
        precisions.append(p)
        recalls.append(r)

    if len(precisions) > 0:
        precisions.insert(0, 0)
        recalls.insert(0, recalls[0])

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    auc = metrics.auc(recalls, precisions)
    # import pdb; pdb.set_trace()
    print('auc = {}'.format(auc))

    if not label:
        label = ''

    label = label + ' ' + '({:.3f})'.format(auc)

    plt.plot(recalls, precisions, label=label, linewidth=4)

    plt.xlabel('recall (fraction of landmarks detected)')
    plt.ylabel('precision (fraction of detections correct)')

    title_template = 'precision vs. recall (error < {} x image width)'
    plt.title(title_template.format(error_threshold))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()


def plot_pr_labelset(imageset_name, source_name_pred, gt_labelset_name, landmarkset_name,
            label=None, epsilon=0.000001, error_threshold=0.01,
            subsample_factor=10, no_conf=False):
    from yogi.db import session

    landmarkset = session.query(LandmarkSet).filter_by(
        name=landmarkset_name).one()

    ground_truth = []
    predictions = []

    for landmark in landmarkset.landmarks:

        landmark_id = landmark.id

        gt = get_labels(labelset_name=gt_labelset_name,
                        imageset_name=imageset_name,
                        landmark_id=landmark_id)

        pred = get_labels(source_name=source_name_pred,
                          imageset_name=imageset_name,
                          landmark_id=landmark_id)

        gt_ids = [label.image_id for label in gt]
        pred_ids = [label.image_id for label in pred]

        common_ids = set(gt_ids).intersection(set(pred_ids))

        gt_dict = dict((label.image_id, label) for label in gt)
        pred_dict = dict((label.image_id, label) for label in pred)

        gt = [gt_dict[image_id] for image_id in common_ids]
        pred = [pred_dict[image_id] for image_id in common_ids]

        ground_truth.extend(gt)
        predictions.extend(pred)

    if no_conf:

        pred = [(label.x, label.y, label.confidence) for label in predictions]
        truth = [(label.x, label.y) for label in ground_truth]

        conf_threshold = None
        p = precision(pred, truth, conf_threshold, error_threshold)
        r = recall(pred, truth, conf_threshold, error_threshold)

        label = label + ' ' + 'P = {:.3f}, R = {:.3f}'.format(p, r)

        plt.scatter([r], [p], label=label)

        plt.xlabel('recall (fraction of landmarks detected)')
        plt.ylabel('precision (fraction of detections correct)')

        title_template = 'precision vs. recall (error < {} x image width)'
        plt.title(title_template.format(error_threshold))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()

        return None

    else:

        conf_values = [label.confidence for label in predictions]
        conf_thresholds = np.unique(conf_values)[:-1]
        conf_thresholds = conf_thresholds[::subsample_factor]

    #    conf_values = [label.confidence for label in predictions]
    #    conf_thresholds = np.unique(conf_values)
    #    conf_thresholds = conf_thresholds[::subsample_factor]

        pred = [(label.x, label.y, label.confidence) for label in predictions]
        truth = [(label.x, label.y) for label in ground_truth]

        print('number of predictions = {}'.format(len(pred)))
        print('number of labels = {}'.format(len(truth)))

        precisions = []
        recalls = []
        for conf_threshold in conf_thresholds:
            p = precision(pred, truth, conf_threshold, error_threshold)
            r = recall(pred, truth, conf_threshold, error_threshold)
            precisions.append(p)
            recalls.append(r)

        if len(precisions) > 0:
            precisions.insert(0, 0)
            recalls.insert(0, recalls[0])

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        auc = metrics.auc(recalls, precisions)
        # import pdb; pdb.set_trace()
        print('auc = {}'.format(auc))

        if not label:
            label = ''

        label = label + ' ' + '({:.3f})'.format(auc)

        plt.plot(recalls, precisions, label=label, linewidth=4)

        plt.xlabel('recall (fraction of landmarks detected)')
        plt.ylabel('precision (fraction of detections correct)')

        title_template = 'precision vs. recall (error < {} x image width)'
        plt.title(title_template.format(error_threshold))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()

        return auc


def subclipset_auc_table(session, subclipset_names, label_source_names,
                         gt_labelset_name, landmarkset_name,
                         threshold_units='normalized', error_threshold=0.05,
                         ci=0.95,
                         use_occluded_gt=True):

    rows = []
    for subclipset_name in subclipset_names:
        for label_source_name in label_source_names:
            print(f'subclipset = {subclipset_name}; source = {label_source_name}')
            auc, plus, minus, values = compute_auc(session,
                              label_source_name, gt_labelset_name,
                              landmarkset_name, threshold_units,
                              error_threshold, use_occluded_gt,
                              ci=ci,
                              subclipset_name=subclipset_name)
   #         for auc_value in values:
   #             row = (subclipset_name, label_source_name, auc, plus, minus, auc_value)
   #             rows.append(row)
            row = (subclipset_name, label_source_name, auc, plus, minus)
            rows.append(row)


    col_names = ('test set', 'model', 'AUC', 'plus', 'minus')  #, 'resampled AUC')
    table = create_table(rows, col_names=col_names)

    return table


#def imageset_auc_table(session,
#                       imageset_names, label_source_names, gt_labelset_name,
#                       landmarkset_name,
#                       threshold_units='normalized', error_threshold=0.05,
#                       ci=0.95,
#                       use_occluded_gt=True):
#
#    rows = []
#    for imageset_name in imageset_names:
#        for label_source_name in label_source_names:
#            auc, plus, minus, values = compute_auc(session,
#                              label_source_name, gt_labelset_name,
#                              landmarkset_name, threshold_units,
#                              error_threshold, use_occluded_gt,
#                              ci=ci,
#                              imageset_name=imageset_name)
#            for auc_value in values:
#                row = (imageset_name, label_source_name, auc, plus, minus, auc_value)
#                rows.append(row)
#
#    col_names = ('test set', 'model', 'AUC', 'plus', 'minus', 'resampled AUC')
#    table = create_table(rows, col_names=col_names)
#
#    return table


def imageset_auc_table(session,
                       imageset_names, label_source_names, gt_labelset_name,
                       landmarkset_name,
                       threshold_units='normalized', error_threshold=0.05,
                       ci=0.95, compute_ci=True,
                       use_occluded_gt=True):

    rows = []
    for imageset_name in imageset_names:
        for label_source_name in label_source_names:
            auc, plus, minus, values = compute_auc(session,
                              label_source_name, gt_labelset_name,
                              landmarkset_name, threshold_units,
                              error_threshold, use_occluded_gt,
                              ci=ci, compute_ci=compute_ci,
                              imageset_name=imageset_name)
            row = (imageset_name, label_source_name, auc, plus, minus)
            rows.append(row)

    col_names = ('test set', 'model', 'AUC', 'plus', 'minus')
    table = create_table(rows, col_names=col_names)
    return table


def auc_barchart(output_path, csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    source_names = list(df['model'])

    fig = plt.figure(figsize=(5, 3))

    labels = [str(auc)[0:4] for auc in df['AUC']]

    plt.bar(x=labels,
        height=df['AUC'],
        yerr=[df['minus'], df['plus']],
        capsize=5.0,
        width=0.8)

    plt.ylim([0.0, 1.0])

    plt.title('AUC')
    plt.savefig(output_path)

    # plt.close(fig)
    plt.close()

def grouped_barplot(df, cat, subcat, val, plus, minus):
    import seaborn as sns
    import pandas as pd
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx)+1.)
    offsets = offsets * (-1)
    width = np.diff(offsets).mean()
    for i, gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.barh(x+offsets[i], dfg[val].values, height=width, 
                label="{} {}".format(subcat, gr), xerr=[dfg[minus].values, dfg[plus].values])
    plt.ylabel(cat)
    plt.xlabel(val)
    plt.xlim(0, 1)
    plt.yticks(x, u)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.gca().yaxis.grid(False)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


def auc_subclipset_barchart(output_path, csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)

    fig = plt.figure(figsize=(15, 15))

    grouped_barplot(df, 'test set', 'model', 'AUC', 'plus', 'minus')
    plt.xlabel('AUC (95% CI)')

    plt.tight_layout()

    plt.savefig(output_path)

    # plt.close(fig)
 

def compute_auc(session, label_source_name, gt_labelset_name,
                landmarkset_name, threshold_units, error_threshold,
                use_occluded_gt, ci=0.95, compute_ci=True, **kwargs):

    predictions, ground_truth = (
        get_comparable_labels(session,
                              label_source_name=label_source_name,
                              gt_labelset_name=gt_labelset_name,
                              landmarkset_name=landmarkset_name,
                              use_occluded_gt=use_occluded_gt,
                              **kwargs))

    print('number of predictions = {}'.format(len(predictions)))
    print('number of labels = {}'.format(len(ground_truth)))
    
    auc = auc_from_labels(predictions, ground_truth,
                          threshold_units=threshold_units,
                          error_threshold=error_threshold)

    predictions = predictions[::5]
    ground_truth = ground_truth[::5]

    if compute_ci:
        replicates = 1000
        auc_values = []
        for i in range(replicates):
            # print(f'replicate {i}')
            (resampled_pred, resampled_gt) = zip(*resample(list(zip(predictions, ground_truth))))
            auc_value = auc_from_labels(resampled_pred, resampled_gt,
                              threshold_units=threshold_units,
                              error_threshold=error_threshold)
            auc_values.append(auc_value)
        ascending_auc_values = sorted(auc_values)
        tail_percentile = (1 - ci) / 2
        lower_idx = int(tail_percentile * replicates)
        lower_idx = max(lower_idx, 0)
        upper_idx = replicates - lower_idx
        upper_idx = min(upper_idx, len(ascending_auc_values) - 1)
        plus = ascending_auc_values[upper_idx] - auc
        minus = auc - ascending_auc_values[lower_idx]
        return (auc, plus, minus, auc_values)
    else:
        return (auc, 0, 0, [])

def create_table(data, col_names):
    import pandas as pd
    df = pd.DataFrame(data, columns=col_names)
    return df


def resample(lst):
    return random.choices(lst, k=len(lst)) 


def get_comparable_labels(session, label_source_name, gt_labelset_name,
                          landmarkset_name, use_occluded_gt, **kwargs):

    landmarkset = session.query(LandmarkSet).filter_by(
        name=landmarkset_name).one()

    ground_truth = []
    predictions = []

    for landmark in landmarkset.landmarks:

        landmark_id = landmark.id

        gt = get_labels(session=session, labelset_name=gt_labelset_name,
                        landmark_id=landmark_id,
                        **kwargs)

        pred = get_labels(session=session, source_name=label_source_name,
                          landmark_id=landmark_id,
                          **kwargs)

        gt_ids = [label.image_id for label in gt
                  if use_occluded_gt or not label.occluded]
        pred_ids = [label.image_id for label in pred]

        common_ids = set(gt_ids).intersection(set(pred_ids))

        gt_dict = dict((label.image_id, label) for label in gt)
        pred_dict = dict((label.image_id, label) for label in pred)

        gt = [gt_dict[image_id] for image_id in common_ids]
        pred = [pred_dict[image_id] for image_id in common_ids]

        ground_truth.extend(gt)
        predictions.extend(pred)

    return (predictions, ground_truth)


def get_imageset_labels(imageset_name, label_source_name, gt_labelset_name,
                        landmarkset_name, use_occluded_gt):

    landmarkset = session.query(LandmarkSet).filter_by(
        name=landmarkset_name).one()

    ground_truth = []
    predictions = []

    for landmark in landmarkset.landmarks:

        landmark_id = landmark.id

        gt = get_labels(labelset_name=gt_labelset_name,
                        imageset_name=imageset_name,
                        landmark_id=landmark_id)

        pred = get_labels(source_name=label_source_name,
                          imageset_name=imageset_name,
                          landmark_id=landmark_id)

        gt_ids = [label.image_id for label in gt
                  if use_occluded_gt or not label.occluded]
        pred_ids = [label.image_id for label in pred]

        common_ids = set(gt_ids).intersection(set(pred_ids))

        gt_dict = dict((label.image_id, label) for label in gt)
        pred_dict = dict((label.image_id, label) for label in pred)

        gt = [gt_dict[image_id] for image_id in common_ids]
        pred = [pred_dict[image_id] for image_id in common_ids]

        ground_truth.extend(gt)
        predictions.extend(pred)

    return (predictions, ground_truth)


def auc_from_labels(predictions, ground_truth,
                    threshold_units='normalized',
                    error_threshold=0.05):

    conf_values = [label.confidence for label in predictions]
    conf_thresholds = np.unique(conf_values)[:-1]

    pred = []
    truth = []
    for (pr, gt) in zip(predictions, ground_truth):
        pred.append(label_to_tuple(pr,
                                   units=threshold_units,
                                   include_conf=True))
        truth.append(label_to_tuple(gt,
                                    units=threshold_units,
                                    include_conf=False))

    precisions = []
    recalls = []
    for (i, conf_threshold) in enumerate(conf_thresholds):
        # print(f'threshold {i}')
        p = precision(pred, truth, conf_threshold, error_threshold)
        r = recall(pred, truth, conf_threshold, error_threshold)
        precisions.append(p)
        recalls.append(r)

    if len(precisions) > 0:
        precisions.insert(0, 0)
        recalls.insert(0, recalls[0])

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    auc = metrics.auc(recalls, precisions)

    return auc


def label_error(pred_label, gt_label, units='normalized'):
    pred_tup = label_to_tuple(pred_label, units=units, include_conf=False)
    gt_tup = label_to_tuple(gt_label, units=units, include_conf=False)
    return np.sqrt((pred_tup[0] - gt_tup[0]) ** 2 +
                   (pred_tup[1] - gt_tup[1]) ** 2)


def label_to_tuple(label, units='normalized', include_conf=True):
    assert(units in ['normalized', 'px', 'digit-length'])

    if units == 'normalized':
        tup = (label.x_norm, label.y_norm)
    elif units == 'px':
        tup = (label.x_px, label.y_px)
    else:
        assert(units == 'digit-length')
        tup = (label.x_digit, label.y_digit)

    if include_conf:
        tup = tup + (label.confidence,)

    return tup
