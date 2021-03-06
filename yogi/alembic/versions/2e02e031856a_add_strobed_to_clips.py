"""add strobed to clips

Revision ID: 2e02e031856a
Revises: 4c7a8b13eae6
Create Date: 2020-03-10 18:02:46.757583

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2e02e031856a'
down_revision = '4c7a8b13eae6'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('clips', schema=None) as batch_op:
        batch_op.add_column(sa.Column('strobed', sa.Boolean(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('clips', schema=None) as batch_op:
        batch_op.drop_column('strobed')

    # ### end Alembic commands ###
