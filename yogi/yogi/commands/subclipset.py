import click
from sqlalchemy.orm.exc import NoResultFound

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import SubClip, SubClipSet, ImageSet
from yogi.utils import handle


def not_found_handler(e):
    print('Could not find subclipset.')


@click.command('create', no_args_is_help=True)
@click.argument('name')
def create_subclipset(name):
    """Create a new empty subclipset."""
    new_subclipset = SubClipSet(name=name)
    session.add(new_subclipset)
    session.commit()


@click.command('add-subclip', no_args_is_help=True)
@click.argument('subclipset_name')
@click.argument('clip_id', type=int)
@click.argument('start_idx')
@click.argument('end_idx')
@handle(NoResultFound, not_found_handler)
def add_subclip(subclipset_name, clip_id, start_idx, end_idx):
    """Add a new subclip to a subclipset."""
    subclipset = session.query(SubClipSet).filter_by(
        name=subclipset_name).one()

    if start_idx == 'None':
        start_idx = None
    else:
        start_idx = int(start_idx)

    if end_idx == 'None':
        end_idx = None
    else:
        end_idx = int(end_idx)

    new_subclip = SubClip(clip_id=clip_id, start_idx=start_idx,
                          end_idx=end_idx)
    session.add(new_subclip)
    session.commit()

    subclipset.subclips.append(new_subclip)
    session.add(subclipset)
    session.commit()


@click.command('delete', no_args_is_help=True)
@click.argument('name')
@handle(NoResultFound, not_found_handler)
def delete_subclipset(name):
    """Delete a subclipset."""
    subclipset = session.query(SubClipSet).filter_by(name=name).one()
    session.delete(subclipset)
    session.commit()


@click.command('show-details', no_args_is_help=True)
@click.argument('subclipset_name')
@handle(NoResultFound, not_found_handler)
def show_subclipset(subclipset_name):
    """Show a subclipset."""
    subclipset = session.query(SubClipSet).filter_by(
        name=subclipset_name).one()
    for subclip in subclipset.subclips:
        print(subclip)


@click.command('to-imageset', no_args_is_help=True)
@click.argument('subclipset_name')
@click.argument('imageset_name')
@handle(NoResultFound, not_found_handler)
def to_imageset(subclipset_name, imageset_name):
    """Convert a subclipset to an imageset."""
    subclipset = session.query(SubClipSet).filter_by(
        name=subclipset_name).one()

    imageset = ImageSet(name=imageset_name)
    session.add(imageset)
    session.commit()

    imageset.images = subclipset.images
    session.add(imageset)
    session.commit()


@click.command('list-all')
def list_subclipsets():
    """List all subclipsets."""
    list_model(SubClipSet, session)


@click.group('subclipset')
def subclipset_command():
    """Create / view / modify subclipsets."""
    pass


subclipset_command.add_command(add_subclip)
subclipset_command.add_command(create_subclipset)
subclipset_command.add_command(to_imageset)
subclipset_command.add_command(show_subclipset)
subclipset_command.add_command(delete_subclipset)
subclipset_command.add_command(list_subclipsets)
