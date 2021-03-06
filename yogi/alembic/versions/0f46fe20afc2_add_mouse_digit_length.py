"""add mouse_digit_length

Revision ID: 0f46fe20afc2
Revises: f06629f6cad9
Create Date: 2020-12-03 17:43:35.705545

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0f46fe20afc2'
down_revision = 'f06629f6cad9'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('clips', schema=None) as batch_op:
        batch_op.add_column(sa.Column('mouse_digit_length', sa.Float(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('clips', schema=None) as batch_op:
        batch_op.drop_column('mouse_digit_length')

    # ### end Alembic commands ###
