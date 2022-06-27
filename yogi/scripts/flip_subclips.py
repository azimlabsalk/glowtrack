from pprint import pprint

from yogi.sql import get_labels
from yogi.models import *
from yogi.db import session

# transfer labels from one subclip to another, and flip them in the x direction

#src_subclipset = session.query(SubClipSet).filter_by(name='challenge-2-graziana-beam-rightward').one()
#dst_subclipset = session.query(SubClipSet).filter_by(name='challenge-2-graziana-beam-rightward-flipped').one()
src_subclipset = session.query(SubClipSet).filter_by(name='challenge-2-keewui-water-side').one()
dst_subclipset = session.query(SubClipSet).filter_by(name='challenge-2-keewui-water-side-flipped').one()

source_name = 'challenge-2-labeled-dan'
source_id = 227

def mirror_landmark_id(landmark_id):
    d = {1:2, 2:1, 3:3}
    return d[landmark_id]

for (src_subclip, dst_subclip) in zip(src_subclipset.subclips, dst_subclipset.subclips):

    print(src_subclip)
    print(dst_subclip)

    src_images = src_subclip.images
    dst_images = dst_subclip.images

    labels = get_labels(clip_id=src_subclip.clip_id, source_name=source_name)

    image_to_labels = {}
    for label in labels:
        lst = image_to_labels.get(label.image_id, [])
        lst.append(label)
        image_to_labels[label.image_id] = lst

    for (src_image, dst_image) in zip(src_images, dst_images):

        assert(src_image.frame_num == dst_image.frame_num)

        if src_image.id in image_to_labels:
            src_labels = image_to_labels[src_image.id]

            for src_label in src_labels:

                dst_image_id = dst_image.id
                dst_source_id = src_label.source_id
                dst_landmark_id = mirror_landmark_id(src_label.landmark_id)
                dst_y = src_label.y
                dst_x = (1 - src_label.x) if (src_label.x is not None) else None
                dst_confidence = src_label.confidence
                dst_occluded = src_label.occluded

                dst_label = Label(image_id=dst_image_id,
                                  source_id=dst_source_id,
                                  landmark_id=dst_landmark_id,
                                  x=dst_x, y=dst_y, confidence=dst_confidence,
                                  occluded=dst_occluded)

                print(src_label)
                print(dst_label)
              
                session.add(dst_label)

            session.commit() 
