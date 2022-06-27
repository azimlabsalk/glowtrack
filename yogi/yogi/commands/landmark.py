import click

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import Landmark


@click.command('list-all')
def list_landmarks():
    """List all landmarks."""
    list_model(Landmark, session)


@click.group('landmark')
def landmark_command():
    """Create / view / modify landmarks."""
    pass


landmark_command.add_command(list_landmarks)
