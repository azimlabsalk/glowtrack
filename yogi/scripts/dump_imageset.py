import sys

from yogi.models import *
from yogi.db import session

name = sys.argv[1]

imageset = session.query(ImageSet).filter_by(name=name).one()
for image in imageset.images:
    print(image.path)

