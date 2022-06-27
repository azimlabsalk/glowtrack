import os

import click
from sqlalchemy.orm.exc import NoResultFound

from yogi import config
from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import LabelSet, Model, LandmarkSet
from yogi.utils import handle
from yogi.nn.models import default_iters, default_scale


def not_found_handler(e):
    print('Could not find model.')


@click.command('create', no_args_is_help=True)
@click.argument('type', type=click.Choice(Model.valid_types))
@click.argument('name')
@click.option('--image-preproc-type',
              type=click.Choice(Model.image_preproc_types), default='none')
@click.option('--training-iters', type=int, default=default_iters)
@click.option('--global-scale', type=float, default=default_scale)
@click.option('--augment-bg', type=bool, default=False)
def create_model(type, name, image_preproc_type, training_iters, global_scale,
                 augment_bg):
    """Create a new model."""

    path = os.path.join(config.models_dir, name)
    os.makedirs(path)

    nn_model = Model(type=type, name=name, path=path, trained=False,
                     image_preproc_type=image_preproc_type,
                     training_iters=training_iters, global_scale=global_scale,
                     augment_bg=augment_bg)

    session.add(nn_model)
    session.commit()


@click.command('copy', no_args_is_help=True)
@click.argument('src_model_name')
@click.argument('dst_model_name')
@click.option('--flip', type=bool, default=False)
@click.option('--test-scale', type=float, default=None)
@click.option('--copydir', type=bool, default=False)
def copy_model(src_model_name, dst_model_name, flip, test_scale, copydir):
    """Copy a model."""
    import shutil

    src_model = session.query(Model).filter_by(name=src_model_name).one()

    src_path = src_model.path
    if copydir:
        dst_path = os.path.join(config.models_dir, dst_model_name)
        shutil.copytree(src_path, dst_path)
    else:
        dst_path = src_path

    if test_scale is None:
        test_scale = src_model.test_scale

    flipped = src_model.flipped
    if flip:
        flipped = not flipped

    dst_model = Model(type=src_model.type,
                      name=dst_model_name,
                      path=dst_path,
                      trained=src_model.trained,
                      image_preproc_type=src_model.image_preproc_type,
                      training_iters=src_model.training_iters,
                      training_set_id=src_model.training_set_id,
                      labelset_id=src_model.labelset_id,
                      global_scale=src_model.global_scale,
                      augment_bg=src_model.augment_bg,
                      test_scale=test_scale,
                      flipped=flipped,
                      landmarkset_id=src_model.landmarkset_id,
                      optimize_scale=src_model.optimize_scale,
                      optimize_scale_fast=src_model.optimize_scale_fast)

    session.add(dst_model)
    session.commit()


@click.command('delete', no_args_is_help=True)
@click.argument('model_name')
@handle(NoResultFound, not_found_handler)
def delete_model(model_name):
    """Delete a model."""
    model = session.query(Model).filter_by(name=model_name).one()
    session.delete(model)
    session.commit()


@click.command('list-all')
def list_models():
    """List all models."""
    list_model(Model, session)


@click.command('show-details', no_args_is_help=True)
@click.argument('model_name')
@handle(NoResultFound, not_found_handler)
def show_model_details(model_name):
    """Show details for a model."""
    model = session.query(Model).filter_by(name=model_name).one()
    print(model)


def train_error_handler(e):
    print('Could not find one of: imageset, model, label source.')
    print(e)


# @click.command('train', no_args_is_help=True)
# @click.argument('model_name')
# @click.argument('imageset_name')
# @click.argument('label_source_name')
# @handle(NoResultFound, train_error_handler)
# def train(model_name, imageset_name, label_source_name):
#     """Train a model on an imageset using given label source."""
#     model = session.query(Model).filter_by(name=model_name).one()
#     imageset = session.query(ImageSet).filter_by(name=imageset_name).one()
#     label_source = session.query(LabelSource).filter_by(
#         name=label_source_name).one()
#
#     print(('Training {} model \'{}\' on imageset \'{}\','
#            'labelsource \'{}\'').format(model.type, model.name,
#                                         imageset.name, label_source.name))
#
#     model.train(imageset, label_source, session, cuda_visible_devices=None)


@click.command('train', no_args_is_help=True)
@click.argument('model_name')
@click.argument('labelset_name')
@click.argument('landmarkset_name')
@click.option('--mirror', type=bool, default=False)
def train(model_name, labelset_name, landmarkset_name, mirror):
    """Train a model on a labelset."""

    model = session.query(Model).filter_by(name=model_name).one()
    labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    landmarkset = session.query(LandmarkSet).filter_by(
        name=landmarkset_name).one()

    print(('Training {} model \'{}\' on labelset \'{}\', landmarkset = {},'
           ' mirroring = {}'
           .format(model.type, model.name, labelset.name, landmarkset.name,
                   mirror)))

    model.train(labelset, landmarkset, session, cuda_visible_devices=None,
                mirror=mirror)


@click.command('export', no_args_is_help=True)
@click.argument('model_name')
@click.argument('json_path')
def export_model(model_name, json_path):
    from yogi.io import export_model
    model = session.query(Model).filter_by(name=model_name).one()
    export_model(model, json_path)


@click.command('import', no_args_is_help=True)
@click.argument('json_path')
def import_model(json_path):
    from yogi.io import import_model
    import_model(json_path)


@click.group('model')
def model_command():
    """Create / view / modify models."""
    pass


model_command.add_command(create_model)
model_command.add_command(copy_model)
model_command.add_command(delete_model)
model_command.add_command(list_models)
model_command.add_command(show_model_details)
model_command.add_command(train)
model_command.add_command(export_model)
model_command.add_command(import_model)
