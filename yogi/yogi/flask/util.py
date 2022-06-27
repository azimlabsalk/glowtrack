import flask
from werkzeug.routing import BaseConverter


class IntListConverter(BaseConverter):
    separator = r'-'
    regex = r'\d+(?:' + separator + r'\d+)*'

    def to_python(self, value):
        return [int(x) for x in value.split(IntListConverter.separator)]

    def to_url(self, value):
        return IntListConverter.separator.join(str(x) for x in value)


def make_image_response(data, type='jpeg'):
    resp = flask.make_response(data)
    resp.content_type = "image/" + type
    return resp


def make_txt_response(data):
    resp = flask.make_response(data)
    resp.content_type = "text/plain"
    return resp

