"""create clip groups

Revision ID: b7d8736e0043
Revises: 10d36b500dd9
Create Date: 2020-06-05 21:44:21.538056

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b7d8736e0043'
down_revision = '10d36b500dd9'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('clip_groups',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('path', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_clip_groups'))
    )
    with op.batch_alter_table('clips', schema=None) as batch_op:
        batch_op.add_column(sa.Column('clip_group_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_clips_clip_group_id'), ['clip_group_id'], unique=False)
        batch_op.create_foreign_key(batch_op.f('fk_clips_clip_group_id_clip_groups'), 'clip_groups', ['clip_group_id'], ['id'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('clips', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('fk_clips_clip_group_id_clip_groups'), type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_clips_clip_group_id'))
        batch_op.drop_column('clip_group_id')

    op.drop_table('clip_groups')
    # ### end Alembic commands ###
