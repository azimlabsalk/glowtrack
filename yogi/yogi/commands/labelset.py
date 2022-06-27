import click
from sqlalchemy.orm.exc import NoResultFound

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import ImageSet, Label, LabelSet, LabelSource
from yogi.sampling import Subsampler
from yogi.utils import handle


def not_found_handler(e):
    print('Could not find labelset.')


@click.command('create-empty', no_args_is_help=True)
@click.argument('name')
def create_empty_labelset(name):
    """Create a new empty labelset."""
    new_labelset = LabelSet(name=name)
    session.add(new_labelset)
    session.commit()


@click.command('create', no_args_is_help=True)
@click.argument('labelset_name')
@click.argument('imageset_name')
@click.argument('label_source_name')
def create_labelset(labelset_name, label_source_name, imageset_name):
    """Create a new labelset from a label source and an imageset."""
    new_labelset = LabelSet(name=labelset_name)
    session.add(new_labelset)
    session.commit()

    label_source = session.query(LabelSource)\
                          .filter_by(name=label_source_name).one()

    imageset = session.query(ImageSet)\
                      .filter_by(name=imageset_name).one()

    labels = Label.labels_for_imageset(session, label_source, imageset)
    new_labelset.labels.extend(labels)
    session.commit()


@click.command('export-coco-gt', no_args_is_help=True)
@click.argument('labelset_name')
@click.argument('file_name')
def export_coco_gt(labelset_name, file_name):
    "Export a labelset to COCO dataset groundtruth format (JSON)."""
    from yogi.coco import export_coco_gt
    import json

    labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    json_data = export_coco_gt(labelset.labels)
    with open(file_name, 'w') as f:
        json.dump(json_data, f)


@click.command('export-coco-results', no_args_is_help=True)
@click.argument('imageset_name')
@click.argument('source_name')
@click.argument('file_name')
def export_coco_results(imageset_name, source_name, file_name):
    "Export labels to COCO dataset results format (JSON)."""
    from yogi.coco import export_coco_results
    from yogi.sql import get_labels
    import json

    labels = get_labels(imageset_name=imageset_name, source_name=source_name)
    json_data = export_coco_results(labels)
    with open(file_name, 'w') as f:
        json.dump(json_data, f)


@click.command('delete', no_args_is_help=True)
@click.argument('name')
@handle(NoResultFound, not_found_handler)
def delete_labelset(name):
    """Delete a labelset."""
    labelset = session.query(LabelSet).filter_by(name=name).one()
    session.delete(labelset)
    session.commit()


@click.command('list-all')
def list_labelsets():
    """List all labelsets."""
    list_model(LabelSet, session)


@click.command('show-details', no_args_is_help=True)
@click.argument('name')
@handle(NoResultFound, not_found_handler)
def show_labelset_details(name):
    """List all labels in a labelset."""
    labelset = session.query(LabelSet).filter_by(name=name).one()
    for label in labelset.labels:
        print(label)


def append_imageset_handler(e):
    print(e)


@click.command('add-imageset', no_args_is_help=True)
@click.argument('labelset_name')
@click.argument('imageset_name')
@click.argument('label_source_name')
@handle(NoResultFound, append_imageset_handler)
def add_imageset(labelset_name, imageset_name, label_source_name):
    """Add an imageset to a labelset."""
    labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    imageset = session.query(ImageSet).filter_by(name=imageset_name).one()
    label_source = session.query(LabelSource)\
                          .filter_by(name=label_source_name).one()

    labels = Label.labels_for_imageset(session, label_source, imageset)
    labelset.labels.extend(labels)
    session.commit()


@click.command('subsample', no_args_is_help=True)
@click.argument('labelset_name')
@click.argument('new_labelset_name')
@click.argument('method', type=click.Choice(Subsampler.valid_types))
@click.argument('fraction', type=float)
@handle(NoResultFound, not_found_handler)
def subsample_labelset(labelset_name, new_labelset_name, method, fraction):
    """Subsample a labelset."""

    labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    subsampler = Subsampler(method=method, fraction=fraction)
    subsample = subsampler.sample(labelset.labels)

    new_labelset = LabelSet(name=new_labelset_name)
    new_labelset.labels.extend(subsample)

    session.add(new_labelset)
    session.commit()


@click.command('extend', no_args_is_help=True)
@click.argument('target', nargs=1)
@click.argument('sources', nargs=-1)
@handle(NoResultFound, not_found_handler)
def extend_labelset(target, sources):
    """Extend a labelset with the contents of other labelsets."""

    target_set = session.query(LabelSet).filter_by(name=target).one()

    for source in sources:
        source_set = session.query(LabelSet).filter_by(name=source).one()
        target_set.labels.extend(source_set.labels)

    session.commit()


@click.group('labelset')
def labelset_command():
    """Create / view / modify labelsets."""
    pass


labelset_command.add_command(add_imageset)
labelset_command.add_command(create_empty_labelset)
labelset_command.add_command(create_labelset)
labelset_command.add_command(delete_labelset)
labelset_command.add_command(subsample_labelset)
labelset_command.add_command(extend_labelset)
labelset_command.add_command(list_labelsets)
labelset_command.add_command(show_labelset_details)
labelset_command.add_command(export_coco_gt)
labelset_command.add_command(export_coco_results)
