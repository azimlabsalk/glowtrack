import click
from sqlalchemy.orm.exc import NoResultFound

from yogi.db import session
from yogi.command_utils import list_model
from yogi.models import AdjMeanSmoother, Smoother
from yogi.utils import handle


def not_found_handler(e):
    print('Could not find smoother.')


@click.command('smooth-clipset', no_args_is_help=True)
@click.argument('smoother_name')
@click.argument('clipset_name')
@click.argument('label_source_name')
def smooth_clipset(smoother_name, clipset_name, label_source_name):
    """Apply a smoother to a clipset."""
    smoother = session.query(Smoother).filter_by(name=smoother_name).one()
    smoother.smooth_clipset(session, clipset_name, label_source_name)


@click.command('smooth-subclipset', no_args_is_help=True)
@click.argument('smoother_name')
@click.argument('subclipset_name')
@click.argument('label_source_name')
def smooth_subclipset(smoother_name, subclipset_name, label_source_name):
    """Apply a smoother to a subclipset."""
    smoother = session.query(Smoother).filter_by(name=smoother_name).one()
    smoother.smooth_subclipset(session, subclipset_name, label_source_name)


@click.command('delete', no_args_is_help=True)
@click.argument('name')
@handle(NoResultFound, not_found_handler)
def delete_smoother(name):
    """Delete a smoother."""
    labelset = session.query(Smoother).filter_by(name=name).one()
    session.delete(labelset)
    session.commit()


@click.command('list-all')
def list_smoothers():
    """List all smoothers."""
    list_model(Smoother, session)


@click.command('create', no_args_is_help=True)
@click.argument('name')
@click.argument('tp_threshold', type=float)
def create_adj_mean_smoother(name, tp_threshold):
    """Create a new 'adjacent mean' smoother."""
    new_smoother = AdjMeanSmoother(name=name, tp_threshold=tp_threshold)
    session.add(new_smoother)
    session.commit()


@click.group('smoother')
def smoother_command():
    """Create / view / modify temporal smoothers."""
    pass


smoother_command.add_command(create_adj_mean_smoother)
smoother_command.add_command(delete_smoother)
smoother_command.add_command(list_smoothers)
smoother_command.add_command(smooth_clipset)
smoother_command.add_command(smooth_subclipset)
