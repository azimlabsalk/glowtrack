from yogi.models import ClipSet, LabelSet, LabelSource
from yogi.db import session
import yogi.sql as sql


red_light_indexes = list(range(0, 4))
white_light_indexes = list(range(4, 9))


def get_labels_for_lights(light_indexes, clipset, label_source):
    all_labels = []
    for light_index in light_indexes:
        labels = sql.get_labels_for_light(light_index, clipset, label_source)
        all_labels.extend(labels)
    return all_labels


def get_labels_for_color(color, clipset, label_source):

    if color == 'white':
        light_indexes = white_light_indexes
    elif color == 'red':
        light_indexes = red_light_indexes
    else:
        raise Exception('"color" should be "red" or "white"')

    labels = get_labels_for_lights(light_indexes, clipset, label_source)

    return labels


if __name__ == '__main__':

    clipset_name = 'cerebro-paired-color'
    label_source_name = 'basic-thresholder'

    clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
    label_source = session.query(LabelSource).filter_by(
        name=label_source_name).one()

    colors = ('red', 'white')
    new_labelsets = ('cerebro-paired-red', 'cerebro-paired-white')

    for (color, new_labelset_name) in zip(colors, new_labelsets):

        new_labelset = LabelSet(name=new_labelset_name)

        session.add(new_labelset)
        session.commit()

        labels = get_labels_for_color(color, clipset, label_source)
        new_labelset.labels = labels

        session.add(new_labelset)
        session.commit()
