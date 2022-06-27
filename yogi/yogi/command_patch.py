import click

original_command = click.command


def command(*args, **kwargs):

    try:
        no_args_is_help = kwargs.pop('no_args_is_help')
    except KeyError:
        no_args_is_help = False

    if no_args_is_help:
        kwargs['cls'] = ShowUsageOnMissingError

    return original_command(*args, **kwargs)


class ShowUsageOnMissingError(click.Command):

    def parse_args(self, ctx, args):
        if not args and not ctx.resilient_parsing:
            click.echo(ctx.get_help(), color=ctx.color)
            ctx.exit()
        else:
            super().parse_args(ctx, args)
