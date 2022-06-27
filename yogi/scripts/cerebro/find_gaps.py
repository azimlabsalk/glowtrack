import numpy as np

from yogi.sql import get_clips_for_labelset, get_labelset_clip

if __name__ == '__main__':
    labelset_name = 'cerebro-all-behaviors-clean'
    clips = get_clips_for_labelset(labelset_name)

    for clip in clips:
        print('processing clip {}'.format(clip.path))

        labels_images = get_labelset_clip(labelset_name, clip.id)
        frame_nums = [image.frame_num for (_, image) in labels_images]

        assert((np.array(frame_nums) == np.arange(len(frame_nums))).all())

        pos = [(label.x, label.y) for (label, _) in labels_images]
        pos = np.array(pos)
        ids = [image.id for (_, image) in labels_images]

        w = labels_images[0][1].width
        h = labels_images[0][1].height
        s = np.array([w, h])

        # UV frames lag visible frames, so interpolate to *next* UV frame

        # check for maximum discrepancy
        diffs = []
        for i in range(pos.shape[0] - 1):
            if ((pos[i, 0] is not None) and
               (pos[i + 1, 0] is not None)):
                diff = np.linalg.norm((pos[i, :] * s) - (pos[i + 1, :] * s))
                diffs.append(diff)

        string = 'max_diff = {:10.2f}, mean_diff = {:10.2f}'
        if len(diffs) == 0:
            print('missing ')
            continue

        print(string.format(np.max(diffs), np.mean(diffs)))

