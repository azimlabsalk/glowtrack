import os

import click
from sqlalchemy.orm.exc import NoResultFound

from yogi import config
from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import ImageSet, LabelSource, Plot, RocCurve
from yogi.utils import handle


def not_found_handler(e):
    print('Could not find plot.')


@click.command('roc-comparison', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('landmarkset_name')
@click.argument('source_name_gt')
@click.argument('source_names_pred', nargs=-1)
@click.option('--subsample-conf', type=int, default=1)
def create_roc_comparison(output_path, imageset_name, landmarkset_name,
                          source_name_gt, source_names_pred,
                          subsample_conf):
    """Create a new ROC comparison plot."""
    from yogi.evaluation import roc_comparison
    roc_comparison(imageset_name, source_names_pred, source_name_gt,
                   landmarkset_name, output_path,
                   subsample_factor=subsample_conf)


@click.command('error-histogram', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('landmarkset_name')
@click.argument('gt_labelset_name')
@click.argument('source_names_pred', nargs=-1)
@click.option('--confidence-thresh', type=float, default=0.5)
def error_histogram(output_path, imageset_name, landmarkset_name,
                         gt_labelset_name, source_names_pred, confidence_thresh):
    """Create a new error histogram plot."""
    from yogi.evaluation import error_histogram_labelset
    error_histogram_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, label=None,
                  epsilon=0.000001, error_threshold=0.05, confidence_thresh=confidence_thresh)


@click.command('pr-comparison', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('landmarkset_name')
@click.argument('source_name_gt')
@click.argument('source_names_pred', nargs=-1)
@click.option('--subsample-conf', type=int, default=1)
def create_pr_comparison(output_path, imageset_name, landmarkset_name,
                         source_name_gt, source_names_pred,
                         subsample_conf):
    """Create a new PR comparison plot."""
    from yogi.evaluation import pr_comparison
    pr_comparison(imageset_name, source_names_pred, source_name_gt,
                  landmarkset_name, output_path,
                  subsample_factor=subsample_conf)


@click.command('pr-comparison-labelset', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('landmarkset_name')
@click.argument('gt_labelset_name')
@click.argument('source_names_pred', nargs=-1)
@click.option('--subsample-conf', type=int, default=1)
@click.option('--no-conf', type=int, default=0)
def create_pr_comparison_labelset(output_path, imageset_name, landmarkset_name,
                         gt_labelset_name, source_names_pred,
                         subsample_conf, no_conf):
    """Create a new PR comparison plot."""
    from yogi.evaluation import pr_comparison_labelset
    pr_comparison_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path,
                  subsample_factor=subsample_conf, no_conf=no_conf)


@click.command('create-auc-barchart', no_args_is_help=True)
@click.argument('output_path')
@click.argument('csv_path')
def create_auc_barchart(output_path, csv_path):
    """Create a new AUC comparison bar chart from an AUC CSV file."""
    from yogi.evaluation import auc_barchart
    auc_barchart(ouptut_path, csv_path)


@click.command('auc-csv', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('landmarkset_name')
@click.argument('gt_labelset_name')
@click.argument('source_names_pred', nargs=-1)
@click.option('--subsample-conf', type=int, default=1)
@click.option('--threshold-units', default="normalized")
@click.option('--error-threshold', type=float, default=0.05)
def auc_csv(output_path, imageset_name, landmarkset_name,
                         gt_labelset_name, source_names_pred,
                         subsample_conf, threshold_units,
                         error_threshold):
    """Compute AUC values with confidence intervals and store in CSV."""
    from yogi.evaluation import imageset_auc_table
    from yogi.db import session
    df = imageset_auc_table(session,
                            imageset_names=[imageset_name],
                            label_source_names=source_names_pred,
                            gt_labelset_name=gt_labelset_name,
                            landmarkset_name=landmarkset_name,
                            threshold_units=threshold_units,
                            error_threshold=error_threshold,
                            use_occluded_gt=True)
    df.to_csv(output_path) 


@click.command('error-cdf-labelset', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('landmarkset_name')
@click.argument('gt_labelset_name')
@click.argument('source_names_pred', nargs=-1)
def create_error_cdf_labelset(output_path, imageset_name, landmarkset_name,
                         gt_labelset_name, source_names_pred):
    """Create a new CDF comparison plot."""
    from yogi.evaluation import error_cdf_labelset
    error_cdf_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path)


@click.command('error-boxplot-labelset', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('landmarkset_name')
@click.argument('gt_labelset_name')
@click.argument('source_names_pred', nargs=-1)
@click.option('--logplot', type=bool, default=True)
@click.option('--ymax', type=int, default=25)
@click.option('--whis', type=bool, default=True)
def create_error_boxplot_labelset(output_path, imageset_name, landmarkset_name,
                         gt_labelset_name, source_names_pred, logplot, ymax,
                         whis):
    """Create a new boxplot comparison plot."""
    from yogi.evaluation import error_boxplot_labelset
    error_boxplot_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path, logplot=logplot, ymax=ymax,
                  whis=whis)


@click.command('error-violinplot-labelset', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('landmarkset_name')
@click.argument('gt_labelset_name')
@click.argument('source_names_pred', nargs=-1)
def create_error_violinplot_labelset(output_path, imageset_name, landmarkset_name,
                         gt_labelset_name, source_names_pred):
    """Create a new violinplot comparison plot."""
    from yogi.evaluation import error_violinplot_labelset
    error_violinplot_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path)


@click.command('error-pvalues-labelset', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('landmarkset_name')
@click.argument('gt_labelset_name')
@click.argument('source_names_pred', nargs=-1)
def create_error_pvalues_labelset(output_path, imageset_name, landmarkset_name,
                         gt_labelset_name, source_names_pred):
    from yogi.evaluation import error_pvalues_labelset
    """Generate p-values comparing pixel error, using KS and AD tests"""
    error_pvalues_labelset(imageset_name, source_names_pred, gt_labelset_name,
                  landmarkset_name, output_path)

@click.command('create-roc', no_args_is_help=True)
@click.argument('imageset_name')
@click.argument('source_name_pred')
@click.argument('source_name_gt')
def create_roc(imageset_name, source_name_pred, source_name_gt):
    """Create a new ROC plot."""
    import uuid

    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()
    source_pred = session.query(LabelSource).filter_by(
        name=source_name_pred).one()
    source_gt = session.query(LabelSource).filter_by(
        name=source_name_gt).one()

    fname = str(uuid.uuid4())
    ext = 'png'
    path = os.path.join(config.roc_dir, fname + '.' + ext)

    roc = RocCurve(path=path, imageset_id=imageset.id,
                   source_id_pred=source_pred.id, source_id_gt=source_gt.id)

    session.add(roc)
    session.commit()

    error_threshold = 0.05
    roc.generate(error_threshold=error_threshold)
    print('generated plot at path "{}"'.format(roc.path))


@click.command('delete', no_args_is_help=True)
@click.argument('plot_name')
@handle(NoResultFound, not_found_handler)
def delete_plot(plot_name):
    """Delete a plot."""
    plot = session.query(Plot).filter_by(name=plot_name).one()
    session.delete(plot)
    session.commit()


@click.command('list-all')
def list_plots():
    """List all plots."""
    list_model(Plot, session)


@click.group('plot')
def plot_command():
    """Create / view / modify plots."""
    pass


plot_command.add_command(create_roc)
plot_command.add_command(delete_plot)
plot_command.add_command(list_plots)
plot_command.add_command(create_roc_comparison)
plot_command.add_command(create_pr_comparison)
plot_command.add_command(create_pr_comparison_labelset)
plot_command.add_command(error_histogram)
plot_command.add_command(create_error_cdf_labelset)
plot_command.add_command(create_error_boxplot_labelset)
plot_command.add_command(create_error_violinplot_labelset)
plot_command.add_command(create_error_pvalues_labelset)
plot_command.add_command(create_auc_barchart)
plot_command.add_command(auc_csv)
