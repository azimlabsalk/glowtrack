import glob
import os
import sys

import click
from flask import abort, Response, Flask, render_template, request, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import cross_origin
from sqlalchemy import or_

from yogi import config
from yogi.models import (Base, Clip, ClipSet, Image,
                         ImageSet, Label, LabelSource, SubClip, SubClipSet,
                         LabelSet, Landmark, LandmarkSet,
                         create_or_update_correction)
from yogi.flask.util import IntListConverter, make_image_response
from yogi.celery.tasks import (clipset_create_async,
                               clipset_create_split_async,
                               clipset_create_split_label_async)


app = Flask(__name__)
app.debug = True

app.url_map.converters['int_list'] = IntListConverter

app.config['SQLALCHEMY_DATABASE_URI'] = config.db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app, model_class=Base)


##### CELERY API ########

@app.route("/api/clipset-create", methods=['POST'])
@cross_origin()
def clipset_create():
    try:
        params = request.get_json()
        clipset_name = params['clipset_name']
        clip_paths = params['clip_paths']
        clipset_create_async(clipset_name, clip_paths)
        return jsonify({'response': f'called clipset_create_async on {clipset_name}'})
    except Exception as e:
        print(e)
        return Response(str(e), status=401)


@app.route("/api/clipset-create-split", methods=['POST'])
@cross_origin()
def clipset_create_split():
    try:
        params = request.get_json()
        clipset_name = params['clipset_name']
        clip_paths = params['clip_paths']
        clipset_create_split_async(clipset_name, clip_paths)
        return jsonify({'response': f'called clipset_create_split_async on {clipset_name}'})
    except Exception as e:
        print(e)
        return Response(str(e), status=401)


@app.route("/api/clipset-create-split-label", methods=['POST'])
@cross_origin()
def clipset_create_split_label():
    try:
        params = request.get_json()
        clipset_name = params['clipset_name']
        clip_paths = params['clip_paths']
        model_name = params['model_name']
        gpu_id = int(params['gpu_id'])
        clipset_create_split_label_async(clipset_name, clip_paths, model_name, gpu_id)
        return jsonify({'response': f'called clipset_create_split_label_async on {clipset_name}'})
    except Exception as e:
        print(e)
        return Response(str(e), status=401)


######### API ###########

@app.route("/api/labels_corrected_csv"
           "/clipset_name/<string:clipset_name>"
           "/source_id/<int:source_id>")
@cross_origin()
def api_labels_corrected_csv(clipset_name, source_id):
    from yogi.sql import get_corrections
    from yogi.sql import get_labels
    from yogi.flask.util import make_txt_response

    labels = get_labels(clip_id=clip_id, source_id=source_id,
                        session=db.session)

    corrections = get_corrections(clip_id=clip_id,
                             original_source_id=source_id,
                             session=db.session)

    corrections_dict = dict([(correction.image_id, correction) for correction in corrections])

    rows = []
    for label in labels:
        corrected_label = corrections_dict.get(label.image_id, label)
        row = (corrected_label.x_px, corrected_label.y_px)
        rows.append(row)

    s = "\n".join([",".join([str(val) for val in row]) for row in rows])

    return make_txt_response(s)


@app.route("/api/labels_corrected_csv"
           "/subclipset_name/<string:subclipset_name>"
           "/source_id/<int:source_id>")
@cross_origin()
def api_labels_corrected_csv_subclips(subclipset_name, source_id):
    from yogi.sql import get_corrections
    from yogi.sql import get_labels
    from yogi.flask.util import make_txt_response

    labels = get_labels(subclip_id=subclip_id, source_id=source_id,
                        session=db.session)

    corrections = get_corrections(clip_id=clip_id,
                             original_source_id=source_id,
                             session=db.session)

    corrections_dict = dict([(correction.image_id, correction) for correction in corrections])

    rows = []
    for label in labels:
        corrected_label = corrections_dict.get(label.image_id, label)
        row = (corrected_label.x_px, corrected_label.y_px)
        rows.append(row)

    s = "\n".join([",".join([str(val) for val in row]) for row in rows])

    return make_txt_response(s)




@app.route("/api/gpus")
@cross_origin()
def api_gpus():
    import nvsmi
    data = [gpu.__dict__ for gpu in nvsmi.get_gpus()]
    return jsonify(data)


@app.route("/api/glob", methods=['POST'])
@cross_origin()
def api_glob():
    try:
        params = request.get_json()
        search = params['search']
        search = search.replace('..', '')

        results = glob.glob(search)
        results = [{"result": result,
                    "parent_dir": parent_dir(result),
                    "permissions": permissions(parent_dir(result))} for result in results]

        print(f'Search: {search}')
        print(f'Results: {results}')
        return jsonify(results)
    except Exception as e:
        print(e)
        return Response(str(e), status=401)


def parent_dir(fname):
    tail, head = os.path.split(fname)
    return tail


def permissions(fname):
    return oct(os.stat(fname).st_mode)[-3:]


@app.route("/api/clipsets")
@cross_origin()
def api_clipsets():
    clipsets = db.session.query(ClipSet).order_by(ClipSet.id.desc()).all()
    data = [clipset.to_dict() for clipset in clipsets]
    return jsonify(data)


@app.route("/api/clipset/<int:id>")
@cross_origin()
def api_clipset(id):
    clipset = db.session.query(ClipSet).filter_by(id=id).one()
    data = clipset.to_dict()
    data['clips'] = [clip.to_dict() for clip in clipset.ordered_clips]
    return jsonify(data)


@app.route("/api/clipset/<int:id>/label_sources")
@cross_origin()
def api_clipset_source_ids(id):
    clipset = db.session.query(ClipSet).filter_by(id=id).one()
    source_ids = clipset.label_source_ids
    sources = [db.session.query(LabelSource).filter_by(id=source_id).one() for source_id in source_ids]
    data = {"label_sources": [{"name": source.name,
                               "id": source.id,
                               #"landmarkset_id": source.landmarkset_id,
                               #"landmarkset_name": source.landmarkset.name
                              }
                              for source in sources]}
    return jsonify(data)


@app.route("/api/subclipsets")
@cross_origin()
def api_subclipsets():
    subclipsets = db.session.query(SubClipSet).order_by(SubClipSet.id.desc()).all()
    data = [subclipset.to_dict() for subclipset in subclipsets]
    return jsonify(data)


@app.route("/api/subclipset/<int:id>")
@cross_origin()
def api_subclipset(id):
    subclipset = db.session.query(SubClipSet).filter_by(id=id).one()
    data = subclipset.to_dict()
    data['subclips'] = [subclip.to_dict() for subclip in subclipset.ordered_subclips]
    return jsonify(data)


@app.route("/api/subclipset/<int:id>/label_sources")
@cross_origin()
def api_subclipset_source_ids(id):
    subclipset = db.session.query(SubClipSet).filter_by(id=id).one()
    source_ids = subclipset.label_source_ids
    sources = [db.session.query(LabelSource).filter_by(id=source_id).one() for source_id in source_ids]
    data = {"label_sources": [{"name": source.name,
                               "id": source.id,
                               #"landmarkset_id": source.landmarkset_id,
                               #"landmarkset_name": source.landmarkset.name
                              }
                              for source in sources]}
    return jsonify(data)


@app.route("/api/create/label", methods=['POST'])
@cross_origin()
def api_create_label():
    try:
        params = request.get_json()

        label = Label(image_id=params['image_id'],
                      landmark_id=params['landmark_id'],
                      source_id=params['source_id'],
                      x=float(params['x']),
                      y=float(params['y']))

        db.session.add(label)
        db.session.commit()
        return Response(status=200)
    except Exception as e:
        return Response(str(e), status=401)


@app.route("/api/update/label", methods=['POST'])
@cross_origin()
def api_update_label():
    try:
        params = request.get_json()

        label = db.session.query(Label).filter_by(id=params['id']).one()

        label.x = float(params['x'])
        label.y = float(params['y'])

        db.session.add(label)
        db.session.commit()
        return Response(status=200)
    except Exception as e:
        return Response(str(e), status=401)


@app.route("/api/create-or-update-correction", methods=['POST'])
@cross_origin()
def create_or_update():
    params = request.get_json()

    print('create-or-update-correction, with params: ' + str(params), flush=True)

    image_id = params['image_id']
    original_source_id = params['original_source_id']
    x = float(params['x'])
    y = float(params['y'])

    create_or_update_correction(image_id, original_source_id, x, y,
                                db.session)

    return Response(status=200)


@app.route("/api/create/clipset", methods=['POST'])
@cross_origin()
def api_create_clipset():
    try:
        params = request.get_json()
        clipset = ClipSet(name=params['clipset_name'])
        db.session.add(clipset)
        db.session.commit()
        return Response(status=200)
    except Exception as e:
        return Response('A clipset already exists with that name', status=401)


@app.route("/api/clip/<int:id>")
@cross_origin()
def api_clip(id):
    clip = db.session.query(Clip).filter_by(id=id).one()
    data = clip.to_dict()
    data['images'] = [image.to_dict() for image in clip.ordered_frames]
    return jsonify(data)


@app.route("/api/subclip/<int:id>")
@cross_origin()
def api_subclip(id):
    subclip = db.session.query(SubClip).filter_by(id=id).one()
    data = subclip.to_dict()
    data['images'] = [image.to_dict() for image in subclip.ordered_frames]
    return jsonify(data)


@app.route("/api/labels"
           "/clip_id/<int:clip_id>"
           "/source_id/<int:source_id>")
@cross_origin()
def api_clip_labels(clip_id, source_id):
    from yogi.sql import get_labels
    labels = get_labels(clip_id=clip_id, source_id=source_id,
                        session=db.session)
    data = [label.to_dict() for label in labels]
    return jsonify(data)


@app.route("/api/labels"
           "/subclip_id/<int:subclip_id>"
           "/source_id/<int:source_id>")
@cross_origin()
def api_subclip_labels(subclip_id, source_id):
    from yogi.sql import get_labels
    labels = get_labels(subclip_id=subclip_id, source_id=source_id,
                        session=db.session)
    data = [label.to_dict() for label in labels]
    return jsonify(data)


@app.route("/api/corrections"
           "/clip_id/<int:clip_id>"
           "/original_source_id/<int:original_source_id>")
@cross_origin()
def api_clip_corrections(clip_id, original_source_id):
    from yogi.sql import get_corrections
    labels = get_corrections(clip_id=clip_id,
                             original_source_id=original_source_id,
                             session=db.session)
    data = [label.to_dict() for label in labels]
    data_dict = dict([(datum['image_id'], datum) for datum in data])
    return jsonify(data_dict)


@app.route("/api/corrections"
           "/subclip_id/<int:subclip_id>"
           "/original_source_id/<int:original_source_id>")
@cross_origin()
def api_subclip_corrections(subclip_id, original_source_id):
    from yogi.sql import get_corrections
    labels = get_corrections(subclip_id=subclip_id,
                             original_source_id=original_source_id,
                             session=db.session)
    data = [label.to_dict() for label in labels]
    data_dict = dict([(datum['image_id'], datum) for datum in data])
    return jsonify(data_dict)


######### MAIN ###########

@app.route("/")
def index():
    clipsets = db.session.query(ClipSet).all()
    imagesets = db.session.query(ImageSet).all()
    labelsets = db.session.query(LabelSet).all()
    html_string = render_template("index.html", clipsets=clipsets,
                                  imagesets=imagesets,
                                  labelsets=labelsets)
    return html_string


@app.route("/clipset_sparklines/<string:clipset_name>/<string:source_name>/<int:landmark_id>")
def clipset_sparklines(clipset_name, source_name, landmark_id):
    clipset = db.session.query(ClipSet).filter_by(
        name=clipset_name).one()
    label_source = db.session.query(LabelSource).filter_by(
        name=source_name).one()

    clips = sorted(clipset.clips, key=lambda clip: clip.id)

    html_string = render_template("clipset_sparklines.html",
                                  clipset=clipset, clips=clips,
                                  landmark_id=landmark_id,
                                  label_source=label_source)
    return html_string


@app.route("/clipset/<int:id>")
def clipset(id):
    clipset = db.session.query(ClipSet).filter_by(id=id).one()
    clips = sort_by(clipset.clips, 'id')
    html_string = render_template("clipset.html", clipset=clipset, clips=clips)
    return html_string


@app.route("/clip/<int:id>")
def clip(id):
    clip = db.session.query(Clip).filter_by(id=id).one()
    images = sort_by(clip.images, 'frame_num')
    html_string = render_template("clip.html", clip=clip, images=images)
    return html_string


@app.route("/sparkline/<int:clip_id>/<int:source_id>"
           "/<int:landmark_id>")
def sparkline(clip_id, source_id, landmark_id):
    import matplotlib
    matplotlib.use('Agg')
    # import numpy as np
    import matplotlib.pyplot as plt
    from io import StringIO
    from yogi.sql import get_labels

    labels = get_labels(clip_id=clip_id, source_id=source_id,
                        landmark_id=landmark_id, session=db.session)

    # labels = np.array([(label.x, label.y, label.confidence)
    #                    for label in labels])

    fig = plt.figure(figsize=(10, 3))
    ax = plt.gca()

    if len(labels) > 0:

        ax.plot([label.confidence for label in labels], color='lightgray',
            label='conf', linestyle=':')
        ax.plot([label.x for label in labels], label='x')
        ax.plot([label.y for label in labels], label='y')
        ax.legend()

        ax.set_ylim(0, 1)

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    plt.close('all')

    imgdata.seek(0)  # rewind the data
    svg_data = imgdata.getvalue()

    return make_image_response(svg_data, type='svg+xml')


@app.route("/subclip_sparkline/<int:subclip_id>/<int:source_id>"
           "/<int:landmark_set_id>")
def subclip_sparkline(subclip_id, source_id, landmark_set_id):
    import matplotlib
    matplotlib.use('Agg')
    # import numpy as np
    import matplotlib.pyplot as plt
    from io import StringIO

    subclip = db.session.query(SubClip).filter_by(id=subclip_id).one()

    landmark_set = db.session.query(LandmarkSet)\
        .filter_by(id=landmark_set_id).one()

    n_landmarks = len(landmark_set.landmarks)

    fig = plt.figure()
    fig, axs = plt.subplots(n_landmarks, 1, sharex=True,
                            figsize=(10, 3 * n_landmarks))

    for i in range(n_landmarks):

        landmark = landmark_set.landmarks[i]
        try:
            ax = axs[i]
        except Exception:
            ax = axs

        query = db.session.query(Label)\
            .filter(Label.source_id == source_id)\
            .filter(Label.landmark_id == landmark.id)\
            .join(Image)\
            .filter(Image.clip_id == subclip.clip_id)\
            .filter(Image.illumination == 'visible')\
            .filter(Label.image_id == Image.id)

        if subclip.start_idx is not None:
            query = query.filter(Image.frame_num >= subclip.start_idx)

        if subclip.end_idx is not None:
            query = query.filter(Image.frame_num < subclip.end_idx)

        labels = query\
            .order_by(Image.frame_num.asc())\
            .all()

        assert(len(labels) > 0)

        ax.plot([label.confidence for label in labels], color='lightgray',
                label='conf', linestyle=':')

        x_label = landmark.name + '-x'
        y_label = landmark.name + '-y'

        ax.plot([label.x for label in labels], label=x_label)
        ax.plot([label.y for label in labels], label=y_label)

        ax.legend()

        ax.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    plt.close('all')

    imgdata.seek(0)  # rewind the data
    svg_data = imgdata.getvalue()

    return make_image_response(svg_data, type='svg+xml')


# compute_auc(label_source_name, gt_labelset_name,
#                 landmarkset_name, threshold_units, error_threshold,
#                 use_occluded_gt, **kwargs):
@app.route("/subclipset_ranked/<string:subclipset_name>/"
           "<string:label_source_name>/"
           "<string:gt_labelset_name>/"
           "<string:landmarkset_name>/"
           "<string:threshold_units>/"
           "<float:error_threshold>/"
           "<string:use_occluded_gt>")
def subclipset_ranked(subclipset_name, label_source_name, gt_labelset_name,
                      landmarkset_name, threshold_units, error_threshold,
                      use_occluded_gt):
    from yogi.evaluation import get_comparable_labels, label_error

    landmark_name = landmarkset_name
    landmark = db.session.query(Landmark).filter_by(name=landmark_name).one()

    predictions, ground_truth = (
        get_comparable_labels(label_source_name=label_source_name,
                              gt_labelset_name=gt_labelset_name,
                              landmarkset_name=landmarkset_name,
                              use_occluded_gt=use_occluded_gt,
                              subclipset_name=subclipset_name,
                              session=db.session))
    # sort by confidence
    data = []
    for (pred_label, gt_label) in zip(predictions, ground_truth):
        datum = {}

        datum['confidence'] = pred_label.confidence

        if not gt_label.is_hidden():
            datum['error'] = label_error(pred_label, gt_label,
                                         units=threshold_units)
            datum['on_target'] = datum['error'] < error_threshold
        else:
            datum['error'] = 'n/a'
            datum['on_target'] = 'n/a'

        datum['occluded'] = gt_label.occluded
        datum['hidden'] = gt_label.is_hidden()

        label_source_ids = [gt_label.source_id, pred_label.source_id]
        datum['image_url'] = url_for('labeled_image_landmark',
                                     id=pred_label.image_id,
                                     landmark_id=landmark.id,
                                     label_source_ids=label_source_ids)

        data.append(datum)

    sorted_data = sorted(data,
                         reverse=True,
                         key=lambda datum: datum['confidence'])

    html_string = render_template("subclipset_ranked.html",
                                  data=sorted_data,
                                  subclipset_name=subclipset_name)

    return html_string


@app.route("/subclipset_sparklines/<string:subclipset_name>/"
           "<string:source_name>/<int:landmark_set_id>")
def subclipset_sparklines(subclipset_name, source_name, landmark_set_id):
    subclipset = db.session.query(SubClipSet).filter_by(
        name=subclipset_name).one()
    label_source = db.session.query(LabelSource).filter_by(
        name=source_name).one()

    subclips = sorted(subclipset.subclips, key=lambda subclip: subclip.id)

    html_string = render_template("subclipset_sparklines.html",
                                  subclipset=clipset, subclips=subclips,
                                  label_source=label_source,
                                  landmark_set_id=landmark_set_id)
    return html_string


@app.route("/subclipset/<int:id>")
def subclipset(id):
    subclipset = db.session.query(SubClipSet).filter_by(id=id).one()
    subclips = sort_by(subclipset.subclips, 'id')
    html_string = render_template("subclipset.html", subclipset=subclipset,
                                  subclips=subclips)
    return html_string


@app.route("/subclip/<int:id>")
def subclip(id):
    subclip = db.session.query(SubClip).filter_by(id=id).one()
    images = subclip.images

    limit = request.args.get('limit', type=int)

    html_string = render_template("subclip.html", subclip=subclip,
                                  images=images, limit=limit)
    return html_string


@app.route("/imageset/<int:id>")
def imageset(id):
    imageset = db.session.query(ImageSet).filter_by(id=id).one()
    try:
        images = sort_by(imageset.images, 'clip_id')
    except TypeError:
        images = imageset.images

    limit = request.args.get('limit', type=int)

    label_source_ids = imageset.label_source_ids
    html_string = render_template("imageset.html", imageset=imageset,
                                  images=images, limit=limit,
                                  label_source_ids=label_source_ids)
    return html_string


@app.route("/subclip_viewer/<int:subclip_id>/<int_list:source_ids>/"
           "<int:frame_num>")
def subclip_viewer(subclip_id, source_ids, frame_num):
    subclip = db.session.query(SubClip).filter_by(id=subclip_id).one()
    image = subclip.images[frame_num]
    next_frame_num = (frame_num + 1 if frame_num < subclip.size - 1
                      else None)
    prev_frame_num = frame_num - 1 if frame_num > 0 else None
    html_string = render_template("subclip_viewer.html", subclip=subclip,
                                  label_source_ids=source_ids,
                                  image=image,
                                  frame_num=frame_num,
                                  next_frame_num=next_frame_num,
                                  prev_frame_num=prev_frame_num)
    return html_string


@app.route("/labelset/<int:id>")
def labelset(id):
    labelset = db.session.query(LabelSet).filter_by(id=id).one()
    limit = request.args.get('limit', type=int)
    html_string = render_template("labelset.html", labelset=labelset,
                                  limit=limit)
    return html_string


@app.route("/image/<int:id>")
def image(id):
    image = db.session.query(Image).filter_by(id=id).one()
    data = open(image.path, 'rb').read()
    return make_image_response(data)


@app.route("/clip_viewer/<int:clip_id>/<int_list:source_ids>/<int:frame_num>")
def clip_viewer(clip_id, source_ids, frame_num):
    clip = db.session.query(Clip).filter_by(id=clip_id).one()
    image = clip.ordered_frames[frame_num]
    next_frame_num = frame_num + 1 if frame_num < clip.num_frames - 1 else None
    prev_frame_num = frame_num - 1 if frame_num > 0 else None
    html_string = render_template("clip_viewer.html", clip=clip,
                                  label_source_ids=source_ids,
                                  image=image,
                                  frame_num=frame_num,
                                  next_frame_num=next_frame_num,
                                  prev_frame_num=prev_frame_num)
    return html_string


@app.route("/labeled_image/<int:id>/<int_list:label_source_ids>")
def labeled_image(id, label_source_ids):
    from yogi.utils import img_to_blob
    from yogi.graphics import render_text

    image = db.session.query(Image).filter_by(id=id).one()
    np_img = image.render_labels(label_source_ids, db.session)

    label_source_area = request.args.get('label_source_area', type=int)
    if label_source_area is not None:
        label = db.session.query(Label)\
            .filter(Label.source_id == label_source_area)\
            .join(Image)\
            .filter(Image.id == id).one()
        render_text(np_img, 20/image.width, 50/image.height,
                    '{:.4f}'.format(label.area))
        render_text(np_img, 20/image.width, 80/image.height,
                    '{:.4f}'.format(label.mean_value))

    data = img_to_blob(np_img)
    return make_image_response(data)


@app.route("/labeled_image_landmark/<int:id>/<int_list:label_source_ids>"
           "/<int:landmark_id>")
def labeled_image_landmark(id, label_source_ids, landmark_id):
    from yogi.utils import img_to_blob
    from yogi.graphics import render_text

    image = db.session.query(Image).filter_by(id=id).one()
    np_img = image.render_labels(label_source_ids, db.session,
                                 landmark_id=landmark_id)

    label_source_area = request.args.get('label_source_area', type=int)
    if label_source_area is not None:
        label = db.session.query(Label)\
            .filter(Label.source_id == label_source_area)\
            .filter(Label.landmark_id == landmark_id)\
            .join(Image)\
            .filter(Image.id == id).one()
        render_text(np_img, 20/image.width, 50/image.height,
                    '{:.4f}'.format(label.area))
        render_text(np_img, 20/image.width, 80/image.height,
                    '{:.4f}'.format(label.mean_value))

    data = img_to_blob(np_img)
    return make_image_response(data)


@app.route("/leaderboard")
def leaderboard():
    html_string = render_template("leaderboard.html")
    return html_string


@app.route("/master_roc")
def master_roc():
    import matplotlib.pyplot as plt
    from yogi.evaluation import plot_roc
    from yogi.models import LabelSource
    # import uuid

    imageset = db.session.query(ImageSet).\
        filter_by(name='bigpaw-random-0.05').one()

    label_source_ids = imageset.label_source_ids
    label_source_ids = [x for x in label_source_ids if x != 4]
    sources = []
    for id in label_source_ids:
        source = db.session.query(LabelSource).filter_by(id=id).one()
        sources.append(source)

    # fname = str(uuid.uuid4())
    fname = 'master_roc'
    ext = 'png'
    roc_path = os.path.join(config.roc_dir, fname + '.' + ext)

    fig = plt.figure()

    for source in sources:
        plot_roc(imageset_name=imageset.name,
                 source_name_pred=source.name,
                 source_name_gt='basic-thresholder',
                 label=source.name,
                 error_threshold=0.05)

    plt.savefig(roc_path)
    plt.close(fig)

    data = open(roc_path, 'rb').read()
    return make_image_response(data)


def sort_by(objs, attr):
    def key(obj):
        return getattr(obj, attr)

    return sorted(objs, key=key)


@click.command('serve')
def serve():
    """Start a webserver to browse Yogi data."""
    app.run(host='0.0.0.0', debug=True)
