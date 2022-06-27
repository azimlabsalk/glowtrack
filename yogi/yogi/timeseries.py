from yogi.db import session
from yogi.models import ClipSet, LandmarkSet
from yogi.sql import get_labels


def get_timeseries_data(clipset_name, predictor_name, gt_name,
                        landmarkset_name):

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()

    landmarkset = session.query(LandmarkSet)\
        .filter_by(name=landmarkset_name).one()

    data = {}

    for clip in clipset.clips:

        clip_data = {
            "width": clip.width,
            "height": clip.height,
            "landmark_data": {}
        }

        for landmark in landmarkset.landmarks:

            pred = get_labels(predictor_name,
                              clip_id=clip.id,
                              landmark_id=landmark.id)

            gt = get_labels(gt_name,
                            clip_id=clip.id,
                            landmark_id=landmark.id)

            assert(len(pred) == len(gt))

            landmark_data = labels_to_dict(pred, gt)
            clip_data["landmark_data"][landmark.name] = landmark_data

        data[clip.id] = clip_data

    return data


def labels_to_dict(pred, gt):
    data = {
        "x_pred": [label.x for label in pred],
        "y_pred": [label.y for label in pred],
        "confidence_pred": [label.confidence for label in pred],
        "x_gt": [label.x for label in gt],
        "y_gt": [label.y for label in gt],
    }
    return data
