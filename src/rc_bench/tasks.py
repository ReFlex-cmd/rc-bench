import logging

from rc_bench.celery_app import celery_app
from rc_bench.config import settings
from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.schema import ExperimentSpec, ResultSpec
from rc_bench.database import get_sync_db_session
from rc_bench.models import Experiment, ExperimentStatus, Result
from rc_bench.runners.pipeline import run_pipeline

logging.getLogger("reservoirpy").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@celery_app.task(name="run_experiment_task")
def run_experiment_task(experiment_id: int) -> None:
    logger.info("[Worker] Starting experiment id=%d", experiment_id)

    with get_sync_db_session() as db:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            logger.error("[Worker] Experiment %d not found in DB", experiment_id)
            return

        experiment.status = ExperimentStatus.RUNNING
        db.commit()

        try:
            spec = ExperimentSpec.model_validate(experiment.config)

            data = get_data_for_experiment(
                dataset_name=spec.dataset.name,
                length=spec.dataset.length,
                train_frac=spec.protocol.train_frac,
                val_frac=spec.protocol.val_frac,
                seed=spec.dataset.seed,
            )

            artifact_dir = settings.ARTIFACT_DIR / str(experiment_id)
            result_spec = run_pipeline(data, spec, artifact_dir=artifact_dir)

            db.add(Result(
                experiment_id=experiment_id,
                result_data=result_spec.model_dump(),
            ))
            experiment.status = ExperimentStatus.COMPLETED
            db.commit()

            if result_spec.multi_seed_result:
                m = result_spec.multi_seed_result
                logger.info(
                    "[Worker] Completed id=%d  mean NRMSE(range)=%.4f  n_seeds=%d",
                    experiment_id, m.mean.nrmse_range, m.n_seeds,
                )
            elif result_spec.metrics:
                logger.info(
                    "[Worker] Completed id=%d  NRMSE(range)=%.4f",
                    experiment_id, result_spec.metrics.nrmse_range,
                )

        except Exception as exc:
            logger.exception("[Worker] Failed experiment id=%d: %s", experiment_id, exc)
            failed = ResultSpec(
                status="failed",
                config_hash="",
                error=str(exc),
            )
            db.add(Result(
                experiment_id=experiment_id,
                result_data=failed.model_dump(),
            ))
            experiment.status = ExperimentStatus.FAILED
            db.commit()
