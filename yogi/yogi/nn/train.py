import os

from sqlalchemy.orm.exc import NoResultFound

from yogi import config
from yogi.db import session
from yogi.models import Model, ImageSet


def train(type, name, imageset_name):

    print('Training {} model \'{}\' on imageset \'{}\''.format(
        type, name, imageset_name))

    try:
        imageset = session.query(ImageSet).filter_by(
            name=imageset_name).one()
    except NoResultFound:
        print('Could not find imageset \'{}\''.format(imageset_name))
        return

    path = os.path.join(config.models_dir, name)
    os.makedirs(path)

    nn_model = Model(name=name, path=path, type=type,
                     imageset_id=imageset.id, trained=False)

    session.add(nn_model)
    session.commit()

    nn_model.train(session, cuda_visible_devices=None)
