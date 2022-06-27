import os

import click
from sqlalchemy.orm.exc import NoResultFound

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import Clip, ClipSet, ImageSet, ClipGroup
from yogi.utils import handle, readlines
from yogi.sampling import Subsampler


def not_found_handler(e):
    print('Could not find clipset.')


@click.command('create-empty', no_args_is_help=True)
@click.argument('name')
def create_empty_clipset(name):
    """Create a new empty clipset."""
    new_clipset = ClipSet(name=name)
    session.add(new_clipset)
    session.commit()


@click.command('create-clips', no_args_is_help=True)
@click.argument('clip_paths', nargs=-1)
@click.argument('clipset_name', nargs=1)
@click.option('--strobed', type=bool, default=True)
@click.option('--flipped', type=bool, default=False)
@click.option('--txt-file')
def create_clips(clip_paths, clipset_name, strobed, txt_file, flipped):
    """Create clips and add them to a clipset.

    By default, CLIP_PATHS are directories that must each contain:

    \b
      uv/video.mp4
      visible/video.mp4

    However if STROBED is False, CLIP_PATHS are just video files.

    """

    if txt_file is not None:
        assert(len(clip_paths) == 0)
        clip_paths = readlines(txt_file)

    try:
        Clip.create_clips(clip_paths, clipset_name, session, strobed=strobed,
                          flipped=flipped)
        session.commit()
    except NoResultFound:
        print('Could not find clipset "{}".'.format(clipset_name))
        if click.confirm('Do you want to create it?'):
            Clip.create_clips(clip_paths, clipset_name, session,
                              strobed=strobed, flipped=flipped,
                              make_set=True)
            session.commit()


@click.command('delete', no_args_is_help=True)
@click.argument('name')
@handle(NoResultFound, not_found_handler)
def delete_clipset(name):
    """Delete a clipset."""
    clipset = session.query(ClipSet).filter_by(name=name).one()
    session.delete(clipset)
    session.commit()


@click.command('add-clip', no_args_is_help=True)
@click.argument('clipset_name')
@click.argument('clip_id')
@handle(NoResultFound, not_found_handler)
def add_clip(clipset_name, clip_id):
    """Add clip to clipset."""
    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    clip = session.query(Clip).filter_by(id=clip_id).one()
    clipset.clips.append(clip)
    session.add(clipset)
    session.commit()


@click.command('group-clips', no_args_is_help=True)
@click.argument('clipset_name')
@handle(NoResultFound, not_found_handler)
def group_clips(clipset_name):
    """Add group clips taken from different cameras."""
    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    clips = clipset.clips
    clipgroup_paths = [clip.path.split('cam')[0] for clip in clips]
    clipgroup_paths = set(clipgroup_paths)
    n_cams = len(set([clip.camera_index for clip in clips]))
    print('n_cams = {}'.format(n_cams))
    assert(len(clipgroup_paths) == len(clipset.clips) / n_cams)

    for (i, clipgroup_path) in enumerate(clipgroup_paths):
        print('{} ({} / {})'.format(clipgroup_path, i, len(clipgroup_paths)))
        clipgroup = ClipGroup(path=clipgroup_path)
        session.add(clipgroup)
        session.commit()

        for clip in clips:
            if clipgroup.path in clip.path:
                print('grouping clip: {}'.format(clip.path))
                clip.clip_group_id = clipgroup.id
                session.add(clip)
        session.commit()


@click.command('list-all')
def list_clipsets():
    """List all clipsets."""
    list_model(ClipSet, session)


@click.command('show-details', no_args_is_help=True)
@click.argument('name')
@handle(NoResultFound, not_found_handler)
def show_clipset_details(name):
    """List all clips in a clipset."""
    clipset = session.query(ClipSet).filter_by(name=name).one()
    for clip in clipset.clips:
        print(clip)


@click.command('split', no_args_is_help=True)
@click.argument('name')
@click.argument('illumination', type=click.Choice(['visible', 'uv']),
                default='visible')
@click.option('--overwrite', type=bool, default=False)
@handle(NoResultFound, not_found_handler)
def clipset_to_frames(name, illumination, overwrite):
    """Convert clipset to individual frames."""
    clipset = session.query(ClipSet).filter_by(name=name).one()
    n_clips = len(clipset.clips)
    for i, clip in enumerate(clipset.clips):
        print('Splitting clip {}/{}  -  {}'.format(
            i+1, n_clips, clip.path))
        clip.video_to_frames(session, illumination=illumination,
                             overwrite=overwrite)


@click.command('detect-dye', no_args_is_help=True)
@click.argument('clipset_name')
@click.argument('detector_name')
@handle(NoResultFound, not_found_handler)
def clipset_centroids(clipset_name, detector_name):
    """Compute dye centroids for a clipset."""
    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    n_clips = len(clipset.clips)
    for i, clip in enumerate(clipset.clips):
        print('Labeling clip {}/{}  -  {}'.format(i+1, n_clips, clip.path))
        clip.compute_centroids(session, detector_name)


@click.command('extend', no_args_is_help=True)
@click.argument('target', nargs=1)
@click.argument('sources', nargs=-1)
@handle(NoResultFound, not_found_handler)
def extend_clipset(target, sources):
    """Extend a clipset with the contents of other clipsets."""

    target_set = session.query(ClipSet).filter_by(name=target).one()

    for source in sources:
        source_set = session.query(ClipSet).filter_by(name=source).one()
        for clip in source_set.clips:
            if clip not in target_set.clips:
                target_set.clips.append(clip)

    session.commit()


@click.command('subtract', no_args_is_help=True)
@click.argument('target', nargs=1)
@click.argument('sources', nargs=-1)
@handle(NoResultFound, not_found_handler)
def subtract_clipset(target, sources):
    """Subtract from a clipset the contents of other clipsets."""

    target_set = session.query(ClipSet).filter_by(name=target).one()

    for source in sources:
        source_set = session.query(ClipSet).filter_by(name=source).one()
        for clip in source_set.clips:
            try:
                target_set.clips.remove(clip)
            except ValueError:
                pass

    session.commit()


@click.command('subsample', no_args_is_help=True)
@click.argument('clipset_name')
@click.argument('new_set_name')
@click.argument('method', type=click.Choice(Subsampler.valid_types))
@click.argument('fraction', type=float)
@click.option('--granularity', type=click.Choice(['clips', 'images']),
              default='images')
@handle(NoResultFound, not_found_handler)
def subsample_clipset(clipset_name, new_set_name, method, fraction,
                      granularity):
    """Subsample each clip in a clipset."""

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    subsampler = Subsampler(method=method, fraction=fraction)

    if granularity == 'images':
        # sample images
        new_imageset = ImageSet(name=new_set_name)
        for clip in clipset.clips:
            subsample = subsampler.sample(clip.images)
            new_imageset.images.extend(subsample)
        session.add(new_imageset)
        session.commit()
    elif granularity == 'clips':
        # sample clips
        new_clipset = ClipSet(name=new_set_name)
        subsample = subsampler.sample(clipset.clips)
        for clip in subsample:
            new_clipset.clips.append(clip)
        session.add(new_clipset)
        session.commit()
    else:
        raise Exception('granularity must be "clips" or "images"')


@click.command('export-labels', no_args_is_help=True)
@click.argument('clipset_name')
@click.argument('label_source_name')
@click.argument('output_dir')
@handle(NoResultFound, not_found_handler)
def export_labels(clipset_name, label_source_name, output_dir):
    """Export labels for a clipset."""
    import numpy as np

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()

    os.makedirs(output_dir, exist_ok=False)

    for clip in clipset.clips:
        labels = clip.get_labels(session, label_source_name, return_array=True)
        filename = 'clip_{}.npy'.format(clip.id)
        output_file = os.path.join(output_dir, filename)
        np.save(output_file, labels)


@click.group('clipset')
def clipset_command():
    """Create / view / modify clipsets."""
    pass


clipset_command.add_command(add_clip)
clipset_command.add_command(group_clips)
clipset_command.add_command(create_empty_clipset)
clipset_command.add_command(create_clips)
clipset_command.add_command(delete_clipset)
clipset_command.add_command(export_labels)
clipset_command.add_command(subsample_clipset)
clipset_command.add_command(list_clipsets)
clipset_command.add_command(show_clipset_details)
clipset_command.add_command(clipset_to_frames)
clipset_command.add_command(clipset_centroids)
clipset_command.add_command(extend_clipset)
clipset_command.add_command(subtract_clipset)
