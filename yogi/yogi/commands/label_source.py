import click

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import LabelSource


@click.command('list-all')
def list_label_sources():
    """List all label sources."""
    list_model(LabelSource, session)


@click.group('label-source')
def label_source_command():
    """Create / view / modify label sources."""
    pass


label_source_command.add_command(list_label_sources)
