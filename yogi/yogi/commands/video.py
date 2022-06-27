import os

import click

from yogi.db import session
from yogi.models import (Clip, ClipSet, ImageSet, LabelSource,
                         SubClipSet, LandmarkSet)
from yogi.video import render_video


@click.command('from-imageset', no_args_is_help=True)
@click.argument('output_path')
@click.argument('imageset_name')
@click.argument('label_source_names', nargs=-1)
@click.option('--landmarkset-name', default=None)
@click.option('--conf-threshold', type=float, default=None)
@click.option('--show-conf', type=bool, default=True)
def video_from_imageset(output_path, imageset_name, label_source_names,
                        landmarkset_name, show_conf, conf_threshold):
    """Create a video from an ImageSet."""

    print("Creating video: ", output_path)

    label_source_ids = []
    for name in label_source_names:
        label_source = session.query(LabelSource).filter_by(name=name).one()
        label_source_ids.append(label_source.id)

    if landmarkset_name is not None:
        landmarkset_id = session.query(LandmarkSet).filter_by(
            name=landmarkset_name).one().id
    else:
        landmarkset_id = None

    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()
    render_video(output_path, imageset.images, label_source_ids, session,
                 conf_threshold=conf_threshold, landmarkset_id=landmarkset_id,
                 quality=10, fps=30, show_conf=show_conf)


@click.command('from-clip', no_args_is_help=True)
@click.argument('output_path')
@click.argument('clip_id')
@click.argument('label_source_names', nargs=-1)
@click.option('--conf-threshold', type=float, default=None)
def video_from_clip(output_path, clip_id, label_source_names, conf_threshold):
    """Create a video from a Clip."""

    print("Creating video: ", output_path)

    label_source_ids = []
    for name in label_source_names:
        label_source = session.query(LabelSource).filter_by(name=name).one()
        label_source_ids.append(label_source.id)

    clip = session.query(Clip).filter_by(id=clip_id).one()
    render_video(output_path, clip.ordered_frames, label_source_ids, session,
                 conf_threshold=conf_threshold, quality=10, fps=30)


@click.command('for-clips', no_args_is_help=True)
@click.argument('clipset_name')
@click.argument('label_source_names', nargs=-1)
@click.option('--conf-threshold', type=float, default=None)
@click.option('--color')
@click.option('--output-dir')
@click.option('--suffix')
def video_for_clips(clipset_name, label_source_names,
                    conf_threshold, color, output_dir, suffix):
    """Create a video for each Clip in a ClipSet."""

    if output_dir is not None:
        print("Creating videos in dir: ", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    suffix = '-labeled.mp4' if suffix is None else suffix + '.mp4'

    label_source_ids = []
    for name in label_source_names:
        label_source = session.query(LabelSource).filter_by(name=name).one()
        label_source_ids.append(label_source.id)

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()

    for clip in clipset.clips:

        if output_dir is not None:
            fname = 'clip-{:06d}.mp4'.format(clip.id)
            output_path = os.path.join(output_dir, fname)
        else:
            base_path = os.path.splitext(clip.path)[0]
            output_path = base_path + suffix
            assert(not os.path.exists(output_path))

        print('creating: {}'.format(output_path))
        render_video(output_path, clip.ordered_frames, label_source_ids,
                     session, conf_threshold=conf_threshold, quality=10,
                     fps=30, color=color)


@click.command('for-subclips', no_args_is_help=True)
@click.argument('subclipset_name')
@click.argument('output-dir')
@click.argument('label_source_names', nargs=-1)
@click.option('--conf-threshold', type=float, default=None)
@click.option('--color')
def video_for_subclips(subclipset_name, output_dir, label_source_names,
                       conf_threshold, color):
    """Create a video for each SubClip in a SubClipSet."""

    if output_dir is not None:
        print("Creating videos in dir: ", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    label_source_ids = []
    for name in label_source_names:
        label_source = session.query(LabelSource).filter_by(name=name).one()
        label_source_ids.append(label_source.id)

    subclipset = session.query(SubClipSet).filter_by(
        name=subclipset_name).one()

    for subclip in subclipset.subclips:

        fname = 'subclip-{:06d}.mp4'.format(subclip.id)
        output_path = os.path.join(output_dir, fname)

        print('creating: {}'.format(output_path))
        render_video(output_path, subclip.images, label_source_ids, session,
                     conf_threshold=conf_threshold, quality=10, fps=30,
                     color=color)


@click.command('for-subclipset', no_args_is_help=True)
@click.argument('subclipset-name')
@click.argument('output-path')
@click.argument('label-source-names', nargs=-1)
@click.option('--conf-threshold', type=float, default=None)
@click.option('--color')
def video_for_subclipset(subclipset_name, output_path, label_source_names,
                         conf_threshold, color):
    """Create a video for a SubClipSet."""

    label_source_ids = []
    for name in label_source_names:
        label_source = session.query(LabelSource).filter_by(name=name).one()
        label_source_ids.append(label_source.id)

    subclipset = session.query(SubClipSet).filter_by(
        name=subclipset_name).one()

    images = []
    for subclip in subclipset.subclips:
        images.extend(subclip.images)

    print('creating: {}'.format(output_path))
    render_video(output_path, images, label_source_ids, session,
                 conf_threshold=conf_threshold, quality=10, fps=30,
                 color=color)


@click.group('video')
def video_command():
    """Create / view / modify videos."""
    pass


video_command.add_command(video_from_imageset)
video_command.add_command(video_from_clip)
video_command.add_command(video_for_clips)
video_command.add_command(video_for_subclips)
video_command.add_command(video_for_subclipset)
