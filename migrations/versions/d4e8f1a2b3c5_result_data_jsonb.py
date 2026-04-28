"""Replace fixed metric columns with result_data JSONB

Revision ID: d4e8f1a2b3c5
Revises: c7f3a1d9e042
Create Date: 2026-04-27 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = 'd4e8f1a2b3c5'
down_revision: Union[str, None] = 'c7f3a1d9e042'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('results', sa.Column('result_data', JSONB(), nullable=True))

    # Migrate existing rows: pack old columns into result_data JSON
    op.execute("""
        UPDATE results SET result_data = jsonb_build_object(
            'status', 'completed',
            'config_hash', '',
            'metrics', jsonb_build_object(
                'nrmse', COALESCE(nrmse, 0),
                'mse',   COALESCE(mse,   0),
                'mae',   COALESCE(mae,   0),
                'val_nrmse',      COALESCE(val_nrmse, 0),
                'execution_time', COALESCE(execution_time, 0)
            ),
            'artifact_paths', COALESCE(meta_data, '{}'::jsonb)
        )
        WHERE result_data IS NULL
    """)

    op.alter_column('results', 'result_data', nullable=False)

    op.drop_column('results', 'nrmse')
    op.drop_column('results', 'mse')
    op.drop_column('results', 'mae')
    op.drop_column('results', 'val_nrmse')
    op.drop_column('results', 'execution_time')
    op.drop_column('results', 'meta_data')


def downgrade() -> None:
    op.add_column('results', sa.Column('nrmse', sa.Float(), nullable=True))
    op.add_column('results', sa.Column('mse', sa.Float(), nullable=True))
    op.add_column('results', sa.Column('mae', sa.Float(), nullable=True))
    op.add_column('results', sa.Column('val_nrmse', sa.Float(), nullable=True))
    op.add_column('results', sa.Column('execution_time', sa.Float(), nullable=True))
    op.add_column('results', sa.Column('meta_data', JSONB(), nullable=True))

    op.execute("""
        UPDATE results SET
            nrmse          = (result_data->'metrics'->>'nrmse')::float,
            mse            = (result_data->'metrics'->>'mse')::float,
            mae            = (result_data->'metrics'->>'mae')::float,
            val_nrmse      = (result_data->'metrics'->>'val_nrmse')::float,
            execution_time = (result_data->'metrics'->>'execution_time')::float,
            meta_data      = result_data->'artifact_paths'
    """)

    op.drop_column('results', 'result_data')
