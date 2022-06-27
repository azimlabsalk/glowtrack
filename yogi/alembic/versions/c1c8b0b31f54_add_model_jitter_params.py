"""add model jitter params

Revision ID: c1c8b0b31f54
Revises: 10d36b500dd9
Create Date: 2020-06-18 13:08:35.362566

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c1c8b0b31f54'
down_revision = '10d36b500dd9'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('scale_jitter_lo', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('scale_jitter_up', sa.Float(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('scale_jitter_up')
        batch_op.drop_column('scale_jitter_lo')

    # ### end Alembic commands ###
