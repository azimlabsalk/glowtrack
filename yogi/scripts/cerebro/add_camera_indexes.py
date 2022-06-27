#!/usr/bin/env python3
from yogi.db import session
from yogi.models import ClipSet

clipset_names = [
    'cerebro-paired',
    'cerebro-paired-nondominant',
    'cerebro-paired-handheld',
    'cerebro-paired-rope',
    'cerebro-paired-free',
]

for clipset_name in clipset_names:

    try:
        clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    except Exception:
        print('could not find ClipSet ' + clipset_name)
        continue

    print('processing ' + clipset_name)

    for clip in clipset.clips:
        clip.camera_index = int(clip.path[-1])

    session.commit()
