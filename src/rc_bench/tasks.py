import logging
from pathlib import Path

import numpy as np
import reservoirpy as rpy
from rc_bench.celery_app import celery_app
from rc_bench.database import get_sync_db_session
from rc_bench.models import Experiment, ExperimentStatus, Result

# --- ИСПОЛЬЗУЕМ АБСОЛЮТНЫЕ ИМПОРТЫ (самые надежные) ---
from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.reservoirs.esn_service import run_esn_experiment
from rc_bench.core.reservoirs.lsm_service import run_lsm_experiment
from rc_bench.core.reservoirs.fhn_service import run_fhn_experiment
from rc_bench.core.reservoirs.logistic_service import run_logistic_experiment

# --- ИСПРАВЛЕНИЕ ОШИБКИ ---
# rpy.verbosity(0) больше не существует в версии 0.4.
# Вместо этого отключаем логи через стандартный logging:
logging.getLogger("reservoirpy").setLevel(logging.ERROR)

@celery_app.task(name="run_experiment_task")
def run_experiment_task(experiment_id: int):
    print(f"🚀 [Worker] Starting Experiment ID: {experiment_id}")
    
    with get_sync_db_session() as db:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            print(f"❌ [Worker] Experiment {experiment_id} not found in DB")
            return

        # 1. Статус RUNNING
        experiment.status = ExperimentStatus.RUNNING
        db.commit()

        try:
            # 2. Получаем данные (в памяти)
            # Если в конфиге нет length, берем 2000 по умолчанию
            length = experiment.config.get("length", 2000)
            seed = experiment.config.get("seed", 42)
            
            data = get_data_for_experiment(
                dataset_name=experiment.dataset_name,
                length=length,
                seed=seed
            )

            # 3. Запускаем вычисления
            rtype = experiment.reservoir_type.lower()
            if rtype == "esn":
                results_data = run_esn_experiment(data, experiment.config)
            elif rtype == "lsm":
                results_data = run_lsm_experiment(data, experiment.config)
            elif rtype == "fhn":
                results_data = run_fhn_experiment(data, experiment.config)
            elif rtype == "logistic":
                results_data = run_logistic_experiment(data, experiment.config)
            else:
                raise ValueError(f"Unsupported reservoir type: {experiment.reservoir_type}")

            # 4. Сохраняем предсказания на диск
            metrics = results_data["metrics"]
            meta = results_data["meta"]

            run_dir = Path("outputs/runs") / str(experiment_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            np.save(run_dir / "preds.npy", results_data["preds"])
            np.save(run_dir / "y_test.npy", results_data["y_test"])

            meta["preds_path"] = str(run_dir / "preds.npy")
            meta["y_test_path"] = str(run_dir / "y_test.npy")

            result = Result(
                experiment_id=experiment_id,
                nrmse=metrics["nrmse"],
                mse=metrics["mse"],
                mae=metrics["mae"],
                val_nrmse=meta.get("val_nrmse"),
                execution_time=metrics["execution_time"],
                meta_data=meta,
            )
            db.add(result)
            
            # 5. Статус COMPLETED
            experiment.status = ExperimentStatus.COMPLETED
            db.commit()
            print(f"✅ [Worker] Success! NRMSE: {metrics['nrmse']:.4f}")
            
        except Exception as e:
            # Логируем полный трейсбэк ошибки в консоль воркера
            import traceback
            traceback.print_exc()
            
            print(f"🔥 [Worker] Failed: {e}")
            experiment.status = ExperimentStatus.FAILED
            db.commit()
