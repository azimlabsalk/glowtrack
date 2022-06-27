import glob

import click

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import AnnotationSet, Annotator, Label, LandmarkSet


@click.command('create-annotation-set', no_args_is_help=True)
@click.argument('json_dir')
@click.argument('landmark_set_id', type=int)
@click.argument('annotation_set_name')
@click.argument('annotator_id', type=int)
def create_annotation_set(json_dir, landmark_set_id,
                          annotation_set_name, annotator_id):
    """Create annotation set from LabelStudio json 'completion' files."""

    annotator = session.query(Annotator).filter_by(id=annotator_id).one()
    landmark_set = session.query(LandmarkSet).filter_by(
        id=landmark_set_id).one()

    annotation_set = AnnotationSet(name=annotation_set_name,
                                   annotator_id=annotator.id,
                                   landmark_set_id=landmark_set.id)

    session.add(annotation_set)
    session.commit()

    json_files = glob.glob(json_dir + '/*')
    annotation_set.add_labels_from_json(session, json_files)


@click.command('import-bounding-boxes', no_args_is_help=True)
@click.argument('json_dir')
@click.argument('annotator_id', type=int)
def import_bounding_boxes(json_dir, annotator_id):
    """Load bounding boxes from LabelStudio json 'completion' files."""
    from yogi.models import BoundingBox

    annotator = session.query(Annotator).filter_by(id=annotator_id).one()

    json_files = glob.glob(json_dir + '/*')
    BoundingBox.create_from_json(session, json_files, annotator_id)


@click.command('delete-set', no_args_is_help=True)
@click.argument('annotation_set_name')
def delete_set(annotation_set_name):
    """Delete an annotation set."""
    annotation_set = session.query(AnnotationSet).filter_by(
        name=annotation_set_name).one()
    labels = session.query(Label).filter_by(source_id=annotation_set.id).all()
    for label in labels:
        session.delete(label)
    session.delete(annotation_set)
    session.commit()


@click.command('merge-annotation-sets', no_args_is_help=True)
@click.argument('set_name_1')
@click.argument('set_name_2')
@click.argument('dst_set_name')
@click.option('--threshold', type=float, default=5.0)
def merge_annotation_sets(set_name_1, set_name_2, dst_set_name, threshold):
    """Create annotation set by merging two others.

    Labels are merged if both source labels are hidden,
    or if they are closer than THRESHOLD pixels apart.
    """
    from yogi.sql import get_labels
    from yogi.annotations import merge_labels

    set1 = session.query(AnnotationSet).filter_by(name=set_name_1).one()
    set2 = session.query(AnnotationSet).filter_by(name=set_name_2).one()

    assert(set1.landmark_set_id == set2.landmark_set_id)
    landmark_set_id = set1.landmark_set_id

    dst_set = session.query(AnnotationSet).filter_by(name=dst_set_name).one()
    assert(dst_set.landmark_set_id == landmark_set_id)

    for landmark in set1.landmark_set.landmarks:

        labels1 = get_labels(set1.name, landmark_id=landmark.id)
        labels2 = get_labels(set2.name, landmark_id=landmark.id)

        print('merging labels for landmark "{}"'.format(landmark.name))
        print('  merging {} labels from {} with {} labels from {}'.format(len(labels1), set1.name, len(labels2), set2.name))

        labels = merge_labels(labels1, labels2, threshold)

        print('  merged {} labels'.format(len(labels)))

        for label in labels:
            label.source_id = dst_set.id
            session.add(label)

        session.commit()


@click.command('list-annotation-sets')
def list_annotation_sets():
    """List all annotation sets."""
    list_model(AnnotationSet, session)


@click.command('list-landmark-sets')
def list_landmark_sets():
    """List all landmark sets."""
    list_model(LandmarkSet, session)


@click.command('list-annotators')
def list_annotators():
    """List all annotators."""
    list_model(Annotator, session)


@click.group('annotation')
def annotation_command():
    """Create / view / modify annotations."""
    pass


annotation_command.add_command(delete_set)
annotation_command.add_command(create_annotation_set)
annotation_command.add_command(list_annotators)
annotation_command.add_command(list_annotation_sets)
annotation_command.add_command(list_landmark_sets)
annotation_command.add_command(merge_annotation_sets)
annotation_command.add_command(import_bounding_boxes)
