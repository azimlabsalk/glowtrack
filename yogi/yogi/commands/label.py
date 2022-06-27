import click

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import ImageSet, Label, Model, SubClipSet, ClipSet, Clip


@click.command('imageset', no_args_is_help=True)
@click.argument('imageset_name')
@click.argument('model_name')
@click.option('--save-scoremaps', type=bool, default=False)
@click.option('--batch-size', type=int, default=100)
def label_imageset(imageset_name, model_name, save_scoremaps, batch_size):
    """Create labels from a model."""
    model = session.query(Model).filter_by(name=model_name).one()
    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()
    model.label_imageset(imageset, session, save_scoremaps=save_scoremaps,
                         commit_batch=batch_size)


@click.command('clipset', no_args_is_help=True)
@click.argument('clipset_name')
@click.argument('model_name')
@click.option('--save-scoremaps', type=bool, default=False)
@click.option('--batch-size', type=int, default=100)
def label_clipset(clipset_name, model_name, save_scoremaps, batch_size):
    """Create labels from a model."""
    model = session.query(Model).filter_by(name=model_name).one()
    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    model.label_clipset(clipset, session, save_scoremaps=save_scoremaps,
                        commit_batch=batch_size)


@click.command('clip', no_args_is_help=True)
@click.argument('clip_id')
@click.argument('model_name')
@click.option('--save-scoremaps', type=bool, default=False)
@click.option('--batch-size', type=int, default=100)
def label_clip(clip_id, model_name, save_scoremaps, batch_size):
    """Create labels from a model."""
    model = session.query(Model).filter_by(name=model_name).one()
    clip = session.query(Clip).filter_by(id=clip_id).one()
    model.label_clip(clip, session, save_scoremaps=save_scoremaps,
                     commit_batch=batch_size)


@click.command('subclipset', no_args_is_help=True)
@click.argument('subclipset_name')
@click.argument('model_name')
@click.option('--save-scoremaps', type=bool, default=False)
@click.option('--batch-size', type=int, default=100)
def label_subclipset(subclipset_name, model_name, save_scoremaps, batch_size):
    """Create labels from a model."""
    model = session.query(Model).filter_by(name=model_name).one()
    subclipset = session.query(SubClipSet).filter_by(
        name=subclipset_name).one()
    model.label_subclipset(subclipset, session, save_scoremaps=save_scoremaps,
                           commit_batch=batch_size)


@click.command('list-all')
def list_labels():
    """List all labels."""
    list_model(Label, session)


@click.group('label')
def label_command():
    """Create / view / modify labels."""
    pass


label_command.add_command(label_clipset)
label_command.add_command(label_subclipset)
label_command.add_command(label_imageset)
label_command.add_command(list_labels)
label_command.add_command(label_clip)
