import os

import click
from sqlalchemy.orm.exc import NoResultFound

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import Clip, ClipSet, Image, ImageSet
from yogi.utils import handle
from yogi.sampling import Subsampler


def not_found_handler(e):
    print('Could not find imageset.')


@click.command('create', no_args_is_help=True)
@click.argument('name')
def create_imageset(name):
    """Create a new empty imageset."""
    new_imageset = ImageSet(name=name)
    session.add(new_imageset)
    session.commit()


@click.command('delete', no_args_is_help=True)
@click.argument('name')
@handle(NoResultFound, not_found_handler)
def delete_imageset(name):
    """Delete a imageset."""
    imageset = session.query(ImageSet).filter_by(name=name).one()
    session.delete(imageset)
    session.commit()


@click.command('list-all')
def list_imagesets():
    """List all imagesets."""
    list_model(ImageSet, session)


@click.command('show-details', no_args_is_help=True)
@click.argument('name')
@handle(NoResultFound, not_found_handler)
def show_imageset_details(name):
    """List all images in an imageset."""
    imageset = session.query(ImageSet).filter_by(name=name).one()
    print(imageset)


def append_clipset_handler(e):
    print('Could not find one of: imageset or clipset.')
    print(e)


@click.command('append-clipset', no_args_is_help=True)
@click.argument('imageset_name')
@click.argument('clipset_name')
@handle(NoResultFound, append_clipset_handler)
def append_clipset(imageset_name, clipset_name):
    """Append a clipset to an imageset."""

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()

    images = clipset.get_images(session)
    imageset.images.extend(images)

    session.commit()


def append_clip_handler(e):
    print('Could not find one of: imageset or clip.')
    print(e)


@click.command('append-clip', no_args_is_help=True)
@click.argument('imageset_name')
@click.argument('clip_id')
@handle(NoResultFound, append_clip_handler)
def append_clip(imageset_name, clip_id):
    """Append a clip to an imageset."""

    clip = session.query(Clip).filter_by(id=clip_id).one()
    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()

    images = clip.images
    imageset.images.extend(images)

    session.commit()


@click.command('add-images', no_args_is_help=True)
@click.argument('imageset_name')
@click.argument('image_paths', nargs=-1)
@handle(NoResultFound, append_clipset_handler)
def add_images(imageset_name, image_paths):
    """Append image files to an imageset."""
    from skimage.io import imread

    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()

    image = imread(image_paths[0])
    h, w = image.shape[0:2]

    images = []
    for image_path in image_paths:
        image_abspath = os.path.realpath(image_path)
        image = Image(path=image_abspath, height=h, width=w)
        images.append(image)
        session.add(image)

    imageset.images.extend(images)
    session.commit()


@click.command('list-label-sources', no_args_is_help=True)
@click.argument('imageset_name')
@handle(NoResultFound, not_found_handler)
def list_label_sources(imageset_name):
    """List all label sources for an imageset."""
    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()
    for label_source in imageset.label_sources:
        print(label_source)


@click.command('subsample', no_args_is_help=True)
@click.argument('imageset_name')
@click.argument('new_imageset_name')
@click.argument('method', type=click.Choice(Subsampler.valid_types))
@click.argument('fraction', type=float)
@handle(NoResultFound, not_found_handler)
def subsample_imageset(imageset_name, new_imageset_name, method, fraction):
    """Subsample an imageset."""

    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()
    subsampler = Subsampler(method=method, fraction=fraction)
    subsample = subsampler.sample(imageset.images)

    new_imageset = ImageSet(name=new_imageset_name)
    new_imageset.images.extend(subsample)

    session.add(new_imageset)
    session.commit()


@click.command('extend', no_args_is_help=True)
@click.argument('target', nargs=1)
@click.argument('sources', nargs=-1)
@handle(NoResultFound, not_found_handler)
def extend_imageset(target, sources):
    """Extend an imageset with the contents of other imagesets."""

    target_set = session.query(ImageSet).filter_by(name=target).one()

    for source in sources:
        source_set = session.query(ImageSet).filter_by(name=source).one()
        target_set.images.extend(source_set.images)

    session.commit()


@click.command('subtract', no_args_is_help=True)
@click.argument('target', nargs=1)
@click.argument('sources', nargs=-1)
@handle(NoResultFound, not_found_handler)
def subtract_imageset(target, sources):
    """Subtract from an imageset the contents of other imagesets."""

    target_set = session.query(ImageSet).filter_by(name=target).one()

    for source in sources:
        source_set = session.query(ImageSet).filter_by(name=source).one()
        for image in source_set.images:
            try:
                target_set.images.remove(image)
            except ValueError:
                pass

    session.commit()


@click.command('intersect', no_args_is_help=True)
@click.argument('set1_name')
@click.argument('set2_name')
@click.argument('output_set_name')
@handle(NoResultFound, not_found_handler)
def intersect_imageset(set1_name, set2_name, output_set_name):
    """Intersect two imagesets and put the result in a new imageset."""

    set1 = session.query(ImageSet).filter_by(name=set1_name).one()
    set2 = session.query(ImageSet).filter_by(name=set2_name).one()

    output_set = ImageSet(name=output_set_name)
    session.add(output_set)

    images1 = set(set1.images)
    images2 = set(set2.images)
    output_images = list(images1.intersection(images2))
    output_set.images.extend(output_images)

    session.commit()


@click.command('export', no_args_is_help=True)
@click.argument('output_file')
@click.argument('imageset_name')
@click.argument('format', type=click.Choice(['label-studio-json']),
                default='label-studio-json')
@click.option('--image-dir', type=str, default=None)
@handle(NoResultFound, append_clipset_handler)
def export(output_file, imageset_name, format, image_dir):
    """Export an imageset."""
    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()

    if format == 'label-studio-json':
        from yogi.label_studio import imageset_to_json
        imageset_to_json(imageset, output_file, image_dir=image_dir)


@click.group('imageset')
def imageset_command():
    """Create / view / modify imagesets."""
    pass


imageset_command.add_command(append_clipset)
imageset_command.add_command(append_clip)
imageset_command.add_command(create_imageset)
imageset_command.add_command(delete_imageset)
imageset_command.add_command(extend_imageset)
imageset_command.add_command(export)
imageset_command.add_command(list_imagesets)
imageset_command.add_command(show_imageset_details)
imageset_command.add_command(subsample_imageset)
imageset_command.add_command(subtract_imageset)
imageset_command.add_command(add_images)
imageset_command.add_command(intersect_imageset)
imageset_command.add_command(list_label_sources)
