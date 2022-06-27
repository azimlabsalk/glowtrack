#!/usr/bin/env python3
from yogi.db import session
from yogi.models import ClipSet, ImageSet, Label, LabelSet, LabelSource


clipset_names = [
    'cerebro-paired-nondominant',
    'cerebro-paired-handheld',
    'cerebro-paired-rope',
    'cerebro-paired-free',
]

basic_thresholder = session.query(LabelSource).filter_by(
    name='basic-thresholder').one()
thresholder5 = session.query(LabelSource).filter_by(name='thresholder-5').one()

for clipset_name in clipset_names:
    print(clipset_name)

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()

    print('creating clipsets')
    try:
        color_clipset = ClipSet(name=clipset_name + '-color')
        session.add(color_clipset)
        mono_clipset = ClipSet(name=clipset_name + '-mono')
        session.add(mono_clipset)
        session.commit()

        for clip in clipset.clips:
            if clip.color_type == 'color':
                color_clipset.clips.append(clip)
            elif clip.color_type == 'mono':
                mono_clipset.clips.append(clip)
        session.commit()
    except Exception as e:
        print(e)

    print('creating imagesets')
    try:
        color_imageset = ImageSet(name=color_clipset.name)
        session.add(color_imageset)
        mono_imageset = ImageSet(name=mono_clipset.name)
        session.add(mono_imageset)
        session.commit()

        color_imageset.images.extend(color_clipset.get_images(session))
        mono_imageset.images.extend(mono_clipset.get_images(session))
        session.commit()
    except Exception as e:
        print(e)

    print('creating labelset')
    try:
        labelset = LabelSet(name=clipset_name)
        session.add(labelset)
        session.commit()

        color_labels = Label.labels_for_imageset(session, basic_thresholder,
                                                 color_imageset)
        labelset.labels.extend(color_labels)

        mono_labels = Label.labels_for_imageset(session, thresholder5,
                                                mono_imageset)
        labelset.labels.extend(mono_labels)

        session.commit()
    except Exception as e:
        print(e)
