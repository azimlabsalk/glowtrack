"""add height and width to Clip and Image

Revision ID: 334da73a79ec
Revises: 195e0334edef
Create Date: 2019-09-30 19:45:11.660399

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '334da73a79ec'
down_revision = '195e0334edef'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('clips', schema=None) as batch_op:
        batch_op.add_column(sa.Column('height', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('width', sa.Integer(), nullable=True))

    with op.batch_alter_table('images', schema=None) as batch_op:
        batch_op.add_column(sa.Column('height', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('width', sa.Integer(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('images', schema=None) as batch_op:
        batch_op.drop_column('width')
        batch_op.drop_column('height')

    with op.batch_alter_table('clips', schema=None) as batch_op:
        batch_op.drop_column('width')
        batch_op.drop_column('height')

    # ### end Alembic commands ###
