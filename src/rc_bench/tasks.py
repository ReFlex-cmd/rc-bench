import logging
from pathlib import Path

import numpy as np
from rc_bench.celery_app import celery_app
from rc_bench.database import get_sync_db_session
from rc_bench.models import Experiment, ExperimentStatus, Result
from rc_bench.core.schema import ExperimentSpec
from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.runners.pipeline import run_pipeline

logging.getLogger("reservoirpy").setLevel(logging.ERROR)


@celery_app.task(name="run_experiment_task")
def run_experiment_task(experiment_id: int):
    print(f"🚀 [Worker] Starting Experiment ID: {experiment_id}")

    with get_sync_db_session() as db:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            print(f"❌ [Worker] Experiment {experiment_id} not found in DB")
            return

        experiment.status = ExperimentStatus.RUNNING
        db.commit()

        try:
            spec = ExperimentSpec.model_validate(experiment.config)

            data = get_data_for_experiment(
                dataset_name=spec.dataset.name,
                length=spec.dataset.length,
                seed=spec.dataset.seed,
            )

            result_spec = run_pipeline(data, spec)

            db.add(Result(
                experiment_id=experiment_id,
                result_data=result_spec.model_dump(),
            ))

            experiment.status = ExperimentStatus.COMPLETED
            db.commit()

            if result_spec.multi_seed_result:
                mean_nrmse = result_spec.multi_seed_result.mean.nrmse_range
                n = result_spec.multi_seed_result.n_seeds
                print(f"✅ [Worker] Success! mean NRMSE(range)={mean_nrmse:.4f} (n={n})")
            elif result_spec.metrics:
                print(f"✅ [Worker] Success! NRMSE(range): {result_spec.metrics.nrmse_range:.4f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"🔥 [Worker] Failed: {e}")

            result_spec = ResultSpec(
                status="failed",
                config_hash="",
                error=str(e),
            )
            db.add(Result(
                experiment_id=experiment_id,
                result_data=result_spec.model_dump(),
            ))
            experiment.status = ExperimentStatus.FAILED
            db.commit()
