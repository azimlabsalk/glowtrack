#!/usr/bin/env python3
from yogi.db import session
from yogi.models import (ClipSet, LabelSet, LabelSource)
import yogi.sql as sql

if __name__ == '__main__':

    clipset_name = 'cerebro-paired-color'

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    label_source = session.query(LabelSource).filter_by(
        name='basic-thresholder').one()

    for light_index in range(9):
        for camera_index in range(8):

    # populate labelset cerebro-1-cam-1-light
    labelset_name = 'cerebro-cam-1-light-test'
    labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    labels = sql.get_labels_for_light_and_camera(light_index, camera_index,
                                                 clipset, label_source)
    labelset.labels = labels
    session.commit()


