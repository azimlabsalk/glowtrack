"""add flipped to model

Revision ID: efd0691aaf5e
Revises: 2e02e031856a
Create Date: 2020-03-13 12:42:13.421695

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'efd0691aaf5e'
down_revision = '2e02e031856a'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('flipped', sa.Boolean(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('flipped')

    # ### end Alembic commands ###