"""add dye-detector erode and size

Revision ID: 10d36b500dd9
Revises: 1939c5c8507f
Create Date: 2020-05-26 16:53:49.050553

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '10d36b500dd9'
down_revision = '1939c5c8507f'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('dye_detectors', schema=None) as batch_op:
        batch_op.add_column(sa.Column('erode', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('size_threshold', sa.Integer(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('dye_detectors', schema=None) as batch_op:
        batch_op.drop_column('size_threshold')
        batch_op.drop_column('erode')

    # ### end Alembic commands ###
