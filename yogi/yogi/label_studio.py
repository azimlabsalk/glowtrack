import json
import os
import urllib.parse

from yogi.models import Label, Image
from yogi.db import session


def imageset_to_json(imageset, output_file, image_dir=None, indent=2):
    import json
    import os
    import urllib
    import shutil

    tasks = {}
    for (i, image) in enumerate(imageset.images):

        if image_dir is not None:
            new_path = os.path.join(image_dir, '{}.png'.format(image.id))
            shutil.copy(image.path, new_path)
            filepath = new_path
        else:
            filepath = image.path

        filename = os.path.basename(filepath)
        task_id = image.id
        params = urllib.parse.urlencode({'d': os.path.dirname(filepath)})
        data_key = 'image'

        base_url = '/'
        image_url_path = base_url + urllib.parse.quote('data/' + filename)
        image_local_url = '{image_url_path}?{params}'.format(
            image_url_path=image_url_path, params=params)

        task = {
            'id': task_id,
#            'task_path': filepath,
            'data': {data_key: image_local_url}
        }

        tasks[str(task_id)] = task

    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=indent)


def load_labels(json_file, landmarks):
    with open(json_file, 'r') as f:
        data = json.load(f)

    labels = []
    for landmark in landmarks:

        label_name = landmark.label_studio_label_name
        checkbox_name = landmark.label_studio_checkbox_name

        (x, y) = (None, None)
        checked = False

        for result in data['completions'][0]['result']:

            if result['type'] == 'keypointlabels':
                if result['value']['keypointlabels'][0] == label_name:
                    (x, y) = (result['value']['x'], result['value']['y'])
                    (x, y) = (x / 100.0, y / 100.0)

            if result['type'] == 'choices':
                checked = (checkbox_name in result['value']['choices'])

        occluded = checked

        #path = get_path_from_url(data['data']['image'])
        #print(path)
        #image = session.query(Image).filter_by(path=path).one()

        image_id = int(data['id'])
        image = session.query(Image).filter_by(id=image_id).one()

        label = Label(x=x, y=y, landmark_id=landmark.id, image_id=image.id,
                      occluded=occluded)
        labels.append(label)

    return labels


def load_bounding_box(json_file):
    from yogi.models import BoundingBox

    with open(json_file, 'r') as f:
        data = json.load(f)

    image_id = int(data['id'])
    try:
        value = data['completions'][0]['result'][0]['value']
        im_width = int(data['completions'][0]['result'][0]['original_width'])
        im_height = int(data['completions'][0]['result'][0]['original_height'])
        x = int(im_width * value['x'] / 100.0)
        y = int(im_height * value['y'] / 100.0)
        width = int(im_width * value['width'] / 100.0)
        height = int(im_height * value['height'] / 100.0)
        bounding_box = BoundingBox(x=x, y=y, width=width, height=height, image_id=image_id)
    except Exception:
        bounding_box = BoundingBox(image_id=image_id)

    return bounding_box


def get_path_from_url(string):
    (url, query_string) = urllib.parse.unquote(string).split('?')
    basepath = urllib.parse.parse_qs(query_string)['d'][0]
    fname = url.split('/')[-1]
    path = os.path.join(basepath, fname)
    return path
