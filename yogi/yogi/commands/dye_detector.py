import click

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import DyeDetector


@click.command('create', no_args_is_help=True)
@click.argument('type', type=click.Choice(DyeDetector.valid_types))
@click.argument('name')
@click.option('--channel', default=0, show_default=True)
@click.option('--threshold', default=30, show_default=True)
def create_imageset(**kwargs):
    """Create a new dye detector."""
    new_dye_detector = DyeDetector(**kwargs)
    session.add(new_dye_detector)
    session.commit()


@click.command('list-all')
def list_dye_detectors():
    """List all dye detectors."""
    list_model(DyeDetector, session)


@click.group('dye-detector')
def dye_detector_command():
    """Create / view / modify dye detectors."""
    pass


dye_detector_command.add_command(create_imageset)
dye_detector_command.add_command(list_dye_detectors)
