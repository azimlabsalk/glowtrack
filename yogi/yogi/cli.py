import click

from yogi.flask.main import serve
from yogi.commands.annotation import annotation_command
from yogi.commands.clip import clip_command
from yogi.commands.clipset import clipset_command
from yogi.commands.dye_detector import dye_detector_command
from yogi.commands.image import image_command
from yogi.commands.imageset import imageset_command
from yogi.commands.labelset import labelset_command
from yogi.commands.label import label_command
from yogi.commands.label_source import label_source_command
from yogi.commands.landmark import landmark_command
from yogi.commands.model import model_command
from yogi.commands.plot import plot_command
from yogi.commands.shell import shell
from yogi.commands.smoother import smoother_command
from yogi.commands.subclipset import subclipset_command
from yogi.commands.video import video_command


@click.group()
def cli():
    """Tools for markerless motion capture."""
    pass


cli.add_command(annotation_command)
cli.add_command(clip_command)
cli.add_command(clipset_command)
cli.add_command(dye_detector_command)
cli.add_command(image_command)
cli.add_command(imageset_command)
cli.add_command(label_command)
cli.add_command(labelset_command)
cli.add_command(label_source_command)
cli.add_command(landmark_command)
cli.add_command(model_command)
cli.add_command(plot_command)
cli.add_command(shell)
cli.add_command(smoother_command)
cli.add_command(subclipset_command)
cli.add_command(serve)
cli.add_command(video_command)
