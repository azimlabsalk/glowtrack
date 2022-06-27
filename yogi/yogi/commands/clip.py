import click
from sqlalchemy.orm.exc import NoResultFound

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import Clip
from yogi.utils import handle


def not_found_handler(e):
    print('Could not find clip.')


@click.command('delete', no_args_is_help=True)
@click.argument('clip_id', type=int)
@handle(NoResultFound, not_found_handler)
def delete_clip(clip_id):
    """Delete a clip."""
    clip = session.query(Clip).filter_by(id=clip_id).one()
    session.delete(clip)
    session.commit()


@click.command('list-all')
def list_clips():
    """List all clips."""
    list_model(Clip, session)


@click.command('show-details', no_args_is_help=True)
@click.argument('clip_id', type=int)
@handle(NoResultFound, not_found_handler)
def show_clip_details(clip_id):
    """Show details for a clip."""
    clip = session.query(Clip).filter_by(id=clip_id).one()
    print(clip)


@click.command('split', no_args_is_help=True)
@click.argument('clip_id', type=int)
@handle(NoResultFound, not_found_handler)
def clip_to_frames(clip_id):
    """Convert clip to individual frames."""
    clip = session.query(Clip).filter_by(id=clip_id).one()
    clip.video_to_frames(session)


@click.command('detect-dye', no_args_is_help=True)
@click.argument('clip_id', type=int)
@click.argument('detector_name', type=str)
@handle(NoResultFound, not_found_handler)
def clip_centroids(clip_id, detector_name):
    """Compute dye centroids for a clip."""
    clip = session.query(Clip).filter_by(id=clip_id).one()
    clip.compute_centroids(session, detector_name)


@click.group('clip')
def clip_command():
    """Create / view / modify clips."""
    pass


clip_command.add_command(delete_clip)
clip_command.add_command(list_clips)
clip_command.add_command(show_clip_details)
clip_command.add_command(clip_to_frames)
clip_command.add_command(clip_centroids)
