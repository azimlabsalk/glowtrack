"""add optimize_scale_fast

Revision ID: 81261cabb991
Revises: 336ebc2a4ead
Create Date: 2020-05-11 13:51:52.644854

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '81261cabb991'
down_revision = '336ebc2a4ead'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('optimize_scale_fast', sa.Boolean(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('optimize_scale_fast')

    # ### end Alembic commands ###
