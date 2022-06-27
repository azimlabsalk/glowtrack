import os
import time

from celery import chain

from yogi.db import session
from yogi.celery.app import app
from yogi.models import Clip, ClipSet, Model


@app.task
def create_clipset(clipset_name, clip_paths):
    Clip.create_clips(clip_paths, clipset_name, session,
                      strobed=False, make_set=True)
    session.commit()


@app.task
def split_clipset(clipset_name):
    clipset = session.query(ClipSet).filter_by(
        name=clipset_name).one()
    for i, clip in enumerate(clipset.clips):
        clip.video_to_frames(session, illumination='visible',
                             overwrite=True)


@app.task
def label_clipset(clipset_name, model_name, gpu_id):
    model = session.query(Model).filter_by(name=model_name).one()
    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    model.label_clipset(clipset, session, save_scoremaps=False,
                        commit_batch=100)


def clipset_create_async(clipset_name, clip_paths):
    create = create_clipset.si(clipset_name, clip_paths)
    pipeline = chain(create)
    pipeline.apply_async()


def clipset_create_split_async(clipset_name, clip_paths):
    create = create_clipset.si(clipset_name, clip_paths)
    split = split_clipset.si(clipset_name)
    pipeline = chain(create, split)
    pipeline.apply_async()


def clipset_create_split_label_async(clipset_name, clip_paths, model_name, gpu_id):
    create = create_clipset.si(clipset_name, clip_paths)
    split = split_clipset.si(clipset_name)
    label = label_clipset.si(clipset_name, model_name, gpu_id)
    pipeline = chain(create, split, label)
    pipeline.apply_async()

