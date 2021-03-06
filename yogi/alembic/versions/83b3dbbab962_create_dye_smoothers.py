"""create dye smoothers

Revision ID: 83b3dbbab962
Revises: 8d93f3862f77
Create Date: 2020-08-31 13:15:33.137966

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '83b3dbbab962'
down_revision = '8d93f3862f77'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('dye_smoothers',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('type', sa.String(length=50), nullable=True),
    sa.ForeignKeyConstraint(['id'], ['smoothers.id'], name=op.f('fk_dye_smoothers_id_smoothers')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_dye_smoothers'))
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('dye_smoothers')
    # ### end Alembic commands ###
