from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rc_bench.core.reservoirs.registry import get_reservoir
from rc_bench.core.schema import (
    ExperimentSpec,
    MetricsResult,
    MetricsSummary,
    MultiSeedResult,
)
from rc_bench.runners.experiment_runner import run_experiment


def run_multi_seed(
    data: Dict[str, Any],
    spec: ExperimentSpec,
    n_seeds: int,
    representative_callback: Optional[
        Callable[[int, Dict[str, Any]], None]
    ] = None,
) -> MultiSeedResult:
    """Run the experiment ``n_seeds`` times, varying only the reservoir seed.

    The dataset and protocol are fixed; only reservoir initialisation differs.
    Seeds used: spec.seed, spec.seed+1, ..., spec.seed+(n_seeds-1).

    If provided, ``representative_callback`` is called exactly once with the
    first seed and its raw run result. This lets callers persist one bounded
    demo artifact without retaining predictions for every seed.
    """
    seeds = [spec.seed + i for i in range(n_seeds)]
    metrics_list: List[MetricsResult] = []

    for index, seed in enumerate(seeds):
        reservoir_config = {"seed": seed, **spec.reservoir.params}
        reservoir = get_reservoir(spec.reservoir.type, reservoir_config)
        result = run_experiment(data, spec, reservoir)
        metrics_list.append(result["metrics"])
        if index == 0 and representative_callback is not None:
            representative_callback(seed, result)

    # Sample std (ddof=1) — variance estimator across seeds. Per audit/03 §3.3.2:
    # methodologically standard for reporting estimator variability in literature.
    # ddof=1 requires n_seeds ≥ 2; for n=1, fall back to 0 to avoid NaN.
    std_ddof = 1 if n_seeds > 1 else 0
    return MultiSeedResult(
        n_seeds=n_seeds,
        seeds=seeds,
        metrics_per_seed=metrics_list,
        mean=MetricsSummary.from_metrics_list(metrics_list, lambda xs: float(np.mean(xs))),
        std=MetricsSummary.from_metrics_list(metrics_list, lambda xs: float(np.std(xs, ddof=std_ddof))),
    )
