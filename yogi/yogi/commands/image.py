import click
from sqlalchemy.orm.exc import NoResultFound

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import Image
from yogi.utils import handle


def not_found_handler(e):
    print('Could not find image.')


@click.command('list-all')
def list_images():
    """List all images."""
    list_model(Image, session)


@click.command('show-details', no_args_is_help=True)
@click.argument('image_id', type=int)
@handle(NoResultFound, not_found_handler)
def show_image_details(image_id):
    """Show details for an image."""
    image = session.query(Image).filter_by(id=image_id).one()
    print(image)


@click.group('image')
def image_command():
    """Create / view / modify images."""
    pass


image_command.add_command(list_images)
image_command.add_command(show_image_details)
