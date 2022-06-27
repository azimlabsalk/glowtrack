import json

from yogi.db import session
from yogi.models import *


def export(obj, fields_to_remove):
    fields = obj.__dict__.copy()
    for field in fields_to_remove:
        fields.pop(field)
    return fields


def export_model(model, json_path):

    json_data = {}

    json_data['model'] = export(model, fields_to_remove=['_sa_instance_state', 'id', 'labelset_id', 'landmarkset_id'])
    
    landmarkset = model.landmarkset
    json_data['landmarkset_name'] = landmarkset.name

    landmarks = []
    for landmark in landmarkset.landmarks:
        landmark_json = export(landmark, fields_to_remove=['_sa_instance_state', 'id', 'mirror_id'])
        landmarks.append(landmark_json)
    json_data['landmarks'] = landmarks
 
    with open(json_path, 'w') as f:
        json.dump(json_data, f)


def import_model(json_path):

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # create landmarks 
    landmarks = []
    for d in json_data['landmarks']:
        landmark = Landmark(**d)
        session.add(landmark) 
        landmarks.append(landmark)

    # create landmarkset
    landmarkset = LandmarkSet(name=json_data['landmarkset_name'])
    session.add(landmarkset)
    landmarkset.landmarks.extend(landmarks)
    session.commit()

    # create model
    model = Model(**json_data['model'])
    model.landmarkset_id = landmarkset.id
    session.add(model)
    session.commit()


