#!/usr/bin/env python3
from yogi.db import session
from yogi.models import (ClipSet, LabelSet, LabelSource)
import yogi.sql as sql

if __name__ == '__main__':

    clipset_name = 'cerebro-paired-color'

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    label_source = session.query(LabelSource).filter_by(
        name='basic-thresholder').one()

    light_index = 4
    camera_index = 3

    # populate labelset cerebro-1-cam-1-light
    labelset_name = 'cerebro-1-cam-1-light'
    labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    labels = sql.get_labels_for_light_and_camera(light_index, camera_index,
                                                 clipset, label_source)
    labelset.labels = labels
    session.commit()

    # populate labelset cerebro-4-cam-1-light
    labelset_name = 'cerebro-4-cam-1-light'
    labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    labels = sql.get_labels_for_light(light_index, clipset, label_source)
    labelset.labels = labels
    session.commit()

    # populate labelset cerebro-4-cam-1-light
    labelset_name = 'cerebro-1-cam-9-light'
    labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    labels = sql.get_labels_for_camera(camera_index, clipset, label_source)
    labelset.labels = labels
    session.commit()
