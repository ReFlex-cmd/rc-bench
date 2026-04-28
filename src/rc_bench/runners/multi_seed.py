from __future__ import annotations

from typing import Any, Dict, List

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
) -> MultiSeedResult:
    """Run the experiment ``n_seeds`` times, varying only the reservoir seed.

    The dataset and protocol are fixed; only reservoir initialisation differs.
    Seeds used: spec.seed, spec.seed+1, ..., spec.seed+(n_seeds-1).
    """
    seeds = [spec.seed + i for i in range(n_seeds)]
    metrics_list: List[MetricsResult] = []

    for seed in seeds:
        reservoir_config = {"seed": seed, **spec.reservoir.params}
        reservoir = get_reservoir(spec.reservoir.type, reservoir_config)
        result = run_experiment(data, spec, reservoir)
        metrics_list.append(result["metrics"])

    return MultiSeedResult(
        n_seeds=n_seeds,
        seeds=seeds,
        metrics_per_seed=metrics_list,
        mean=MetricsSummary.from_metrics_list(metrics_list, lambda xs: float(np.mean(xs))),
        std=MetricsSummary.from_metrics_list(metrics_list, lambda xs: float(np.std(xs, ddof=0))),
    )
