#!/usr/bin/env python3
from yogi.db import session
from yogi.models import ClipSet


clipset_names = [
    'cerebro-paired-nondominant',
    'cerebro-paired-handheld',
    'cerebro-paired-rope',
    'cerebro-paired-free',
    'cerebro-paired-free-calib'
]

for clipset_name in clipset_names:

    print('colorizing clipset: ' + clipset_name)

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()

    for clip in clipset.clips:
        if int(clip.path[-1]) < 4:
            clip.color_type = 'color'
        else:
            clip.color_type = 'mono'

    session.commit()
