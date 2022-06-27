import code
import pprint

import click

import yogi.db
import yogi.models
import yogi.sql


@click.command()
def shell():
    """Open Python prompt with db classes and session variable loaded."""
    local = {}
    for module in [yogi.db, yogi.models, yogi.sql, pprint]:
        local.update(module.__dict__)

    code.interact(local=local)
