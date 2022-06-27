import base64
import json
from textwrap import dedent as d
from io import BytesIO
from PIL import Image
import sys

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import imageio
import pandas as pd

csv_file = sys.argv[1]

# external_stylesheets = []
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv(csv_file, index_col=0)

def df_to_dash(df):
    videos = df.video.unique()
    traces = []
    for video in videos:
        video_df = df[df['video'] == video]
        trace_dict = {}
        trace_dict['name'] = video
        trace_dict['x'] = list(video_df.x)
        trace_dict['y'] = list(video_df.y)
        trace_dict['mode'] = 'markers'
        trace_dict['marker'] = {'size': 4}
        trace_dict['customdata'] = list(video_df.index)
        traces.append(trace_dict)
    return traces


def get_video_frame(video, frame):
    reader = imageio.get_reader(video)
    image = reader.get_data(frame)
    return image


def image_to_base64(img):
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return image_string


data = df_to_dash(df)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div(
    className='row',
    children=[
        dcc.Graph(
            id='basic-interactions',
            className="nine columns",
            figure={
                'data': data,
                'layout': {
                    'clickmode': 'event+select',
                    'height': 600,
                }
            }
        ),
        html.Img(
            id="body-image",
            className='three columns',
        )
    ]
)


@app.callback(Output("body-image", "src"),
             [Input("basic-interactions", "hoverData")])
def update_body_image(hover_data):
    if hover_data is None:
        return ''
    points = hover_data["points"]
    if len(points) > 0:
        index = points[0]["customdata"]
        row = df.ix[index]
        img = get_video_frame(row.video, row.frame)
        img_base64 = image_to_base64(img)
        src = 'data:image/jpeg;base64,' + img_base64
    else:
        src = ''
    return src


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

