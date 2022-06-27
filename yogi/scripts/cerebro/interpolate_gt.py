import numpy as np

from yogi.db import session
from yogi.models import Label, LabelSet, SmoothedLabelSource  #, SmoothedDyeDetector
from yogi.sql import get_clips_for_labelset, get_labelset_clip


def copy_label(label):
    new_label = Label(x=label.x, y=label.y, source_id=label.source_id,
                      image_id=label.image_id, landmark_id=label.landmark_id)
    return new_label


if __name__ == '__main__':

    labelset_name = 'cerebro-all-behaviors-clean'
    labelset = session.query(LabelSet).filter_by(name=labelset_name).one()
    clips = get_clips_for_labelset(labelset_name)

    smoother_type = 'linear'

    for clip in clips:

        if clip.frame_rate != 100:
            continue

        print('processing clip {}'.format(clip.path))

        labels_images = get_labelset_clip(labelset_name, clip.id)

        if not clip.uv_first:
            labels_images = list(reversed(labels_images))

        frame_nums = [image.frame_num for (_, image) in labels_images]

        first_label = labels_images[0][0]
        smoother_id = 5
        source = SmoothedLabelSource.find_or_create(session, first_label.source_id, smoother_id)
        print('creating labels with source: {}'.format(str(source)))
        source_id = source.id

        if clip.uv_first:
            assert((np.array(frame_nums) == np.arange(len(frame_nums))).all())
        else:
            assert((np.array(frame_nums) == np.arange(len(frame_nums))[::-1]).all())


        pos = [(label.x, label.y) for (label, _) in labels_images]
        pos = np.array(pos)
        ids = [image.id for (_, image) in labels_images]

        w = labels_images[0][1].width
        h = labels_images[0][1].height
        s = np.array([w, h])

        new_labels = []

        if smoother_type == 'linear':
            # UV frames lag visible frames, so interpolate to *next* UV frame
            for i in range(pos.shape[0]):

                if pos[i, 0] is None:
                    new_label = copy_label(labels_images[i][0])
                    new_label.source_id = source_id
                    new_labels.append(new_label)
 
                elif (i < pos.shape[0] - 1 and (pos[i + 1, 0] is not None)):
                    new_pos = (pos[i, :] + pos[i + 1, :]) / 2
                    new_label = copy_label(labels_images[i][0])
                    (new_label.x, new_label.y) = new_pos
                    new_label.source_id = source_id 
                    new_labels.append(new_label)

        else:
            msg = 'smoother type "{}" not recognized'.format(smoother_type)
            raise Exception(msg)

        for i in range(len(new_labels)):
            session.add(new_labels[i])
            if (i + 1) % 100 == 0:
                session.commit()

        session.commit()
