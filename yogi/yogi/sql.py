# from yogi.db import session
from yogi.models import (Clip, ClipSet, clipset_association_table, Image, ImageSet,
                         Label, LabelSet, labelset_association_table,
                         subclipset_association_table, SubClip, SubClipSet,
                         LabelSource, CorrectedLabelSource, get_corrected_source)
from sqlalchemy import or_


def get_corrections(original_source_id=None, **kwargs):

    assert(original_source_id is not None)
    assert('session' in kwargs)
    session = kwargs['session']

    source = get_corrected_source(original_source_id, session)
    if source is not None:
        return get_labels(source_id=source.id, **kwargs)
    else:
        return []


def get_labels(source_name=None, source_id=None, imageset_name=None,
               landmark_id=None, clip_id=None, subclip_id=None, return_array=False,
               clipset_name=None,
               labelset_name=None, subclipset_name=None, illumination=None,
               session=None, image_id=None):
    from yogi.db import session as default_session

    session = session or default_session

    query = \
        session.query(Label).\
        select_from(Image, Label).\
        filter(Image.id == Label.image_id)

    if image_id is not None:
        query = query.filter(Image.id == image_id)

    if clip_id is not None:
        query = query.filter(Image.clip_id == clip_id)

    if imageset_name is not None:
        query = query.join(ImageSet.images).\
            filter(ImageSet.name == imageset_name)

    if source_name is not None:
        query = query.join(LabelSource, LabelSource.id == Label.source_id).\
            filter(LabelSource.name == source_name)

    if source_id is not None:
        query = query.join(LabelSource, LabelSource.id == Label.source_id).\
            filter(LabelSource.id == source_id)

    if labelset_name is not None:
        query = query.join(LabelSet.labels).\
            filter(LabelSet.name == labelset_name)

    if subclipset_name is not None:
        query = query.join(Clip)\
            .join(SubClip)\
            .filter(or_(Image.frame_num >= SubClip.start_idx,
                        SubClip.start_idx == None))\
            .filter(or_(Image.frame_num < SubClip.end_idx,
                        SubClip.end_idx == None))\
            .join(subclipset_association_table)\
            .join(SubClipSet)\
            .filter(SubClipSet.name == subclipset_name)

    if subclip_id is not None:
        query = query.join(Clip)\
            .join(SubClip)\
            .filter(or_(Image.frame_num >= SubClip.start_idx,
                        SubClip.start_idx == None))\
            .filter(or_(Image.frame_num < SubClip.end_idx,
                        SubClip.end_idx == None))\
            .filter(SubClip.id == subclip_id)

    if clipset_name is not None:
        query = query.join(Clip)\
            .join(clipset_association_table)\
            .join(ClipSet)\
            .filter(ClipSet.name == clipset_name)

    if illumination is not None:
        query = query.filter(Image.illumination == illumination)

    query = query.order_by(Image.frame_num)

    if landmark_id is not None:
        query = query.filter(Label.landmark_id == landmark_id)

    labels = query.all()

    if return_array:
        import numpy as np
        labels = np.array([(label.x, label.y, label.confidence)
                           for label in labels])

    return labels


def get_labels_for_light_and_camera(light_index, camera_index, clipset,
                                    label_source):
    labels = session.query(Label)\
        .filter(Label.source_id == label_source.id)\
        .join(Image)\
        .filter(Image.light_index == light_index)\
        .join(Clip)\
        .filter(Clip.camera_index == camera_index)\
        .join(clipset_association_table)\
        .filter(clipset_association_table.c.clipset_id == clipset.id)\
        .all()
    return labels


def get_labels_for_light(light_index, clipset, label_source):
    labels = session.query(Label)\
        .filter(Label.source_id == label_source.id)\
        .join(Image)\
        .filter(Image.light_index == light_index)\
        .join(Clip)\
        .join(clipset_association_table)\
        .filter(clipset_association_table.c.clipset_id == clipset.id)\
        .all()
    return labels


def get_clips_for_labelset(labelset_name):
    query = \
        session.query(Clip)\
        .join(Image)\
        .join(Label)\
        .join(labelset_association_table)\
        .join(LabelSet)\
        .filter(LabelSet.name == labelset_name)
    clips = query.all()
    return clips


def get_labelset_clip(labelset_name, clip_id):
    query = \
        session.query(Label, Image)\
        .join(Image)\
        .join(Clip)\
        .filter(Clip.id == clip_id)\
        .join(labelset_association_table)\
        .join(LabelSet)\
        .filter(LabelSet.name == labelset_name)\
        .order_by(Image.frame_num)
    labels = query.all()
    return labels


def get_labels_for_camera(camera_index, clipset, label_source):
    labels = session.query(Label)\
        .filter(Label.source_id == label_source.id)\
        .join(Image)\
        .join(Clip)\
        .filter(Clip.camera_index == camera_index)\
        .join(clipset_association_table)\
        .filter(clipset_association_table.c.clipset_id == clipset.id)\
        .all()
    return labels


# def get_labels_for_labelset_and_clip(labelset_name, clip_id):
#     labels = session.query(Label)\
#         .join(labelset_association_table)
#         .join(LabelSet)
#         .filter(LabelSet.name == 'cerebro-clipgroup-test')
#         .filter(Label.source_id == label_source.id)\
#         .join(Image)\
#         .filter(Image.light_index == light_index)\
#         .join(Clip)\
#         .filter(Clip.camera_index == camera_index)\
#         .join(clipset_association_table)\
#         .filter(clipset_association_table.c.clipset_id == clipset.id)\
#         .all()
#     return labels


def update_landmark_id(labels, landmark_id):
    from sqlalchemy.sql import bindparam
    from yogi.db import engine
    from yogi.models import Label
    table = Label.__table__
    stmt = table.update().\
        where(table.c.id == bindparam('id_')).\
        values(landmark_id=bindparam('new_landmark_id'))
    update_dicts = [{'id_': label.id, 'new_landmark_id': landmark_id}
                    for label in labels]
    conn = engine.connect()
    conn.execute(stmt, update_dicts)

# from sqlalchemy.orm import aliased
#
# def get_labels(imageset_name, source1_name, source2_name):
#     label1 = aliased(Label)
#     label2 = aliased(Label)
#     labelsource1 = aliased(LabelSource)
#     labelsource2 = aliased(LabelSource)
#     labeled_images = \
#         session.query(label1, label2).\
#         select_from(Image, label1, label2).\
#         join(ImageSet.images).\
#         filter(ImageSet.name == imageset_name).\
#         filter(Image.id == label1.image_id).\
#         filter(Image.id == label2.image_id).\
#         join(labelsource1, labelsource1.id == label1.source_id).\
#         filter(labelsource1.name == source1_name).\
#         join(labelsource2, labelsource2.id == label2.source_id).\
#         filter(labelsource2.name == source2_name).\
#         all()
#     return labeled_images
