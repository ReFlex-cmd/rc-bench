"""Add mse, val_nrmse to results; make owner_id NOT NULL

Revision ID: c7f3a1d9e042
Revises: a253bc84cc78
Create Date: 2026-04-07 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7f3a1d9e042'
down_revision: Union[str, Sequence[str], None] = 'a253bc84cc78'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Новые колонки в results
    op.add_column('results', sa.Column('mse', sa.Float(), nullable=True))
    op.add_column('results', sa.Column('val_nrmse', sa.Float(), nullable=True))

    # owner_id: заполняем NULL-записи дефолтным пользователем, затем NOT NULL
    op.execute("""
        UPDATE experiments
        SET owner_id = (SELECT id FROM users ORDER BY id LIMIT 1)
        WHERE owner_id IS NULL
    """)
    op.alter_column('experiments', 'owner_id', nullable=False)


def downgrade() -> None:
    op.alter_column('experiments', 'owner_id', nullable=True)
    op.drop_column('results', 'val_nrmse')
    op.drop_column('results', 'mse')
