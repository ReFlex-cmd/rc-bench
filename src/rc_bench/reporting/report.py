"""Generate Markdown and CSV reports from a collection of RunRecord objects."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from rc_bench.reporting.run_record import RunRecord

_METRIC_COLS = [
    "nrmse_range",
    "nrmse_std",
    "nrmse_var",
    "rmse",
    "mae",
    "mse",
    "prediction_horizon",
    "val_nrmse_range",
    "train_time",
    "inference_latency",
    "peak_memory",
]


def _get_metrics_dict(record: RunRecord) -> Dict[str, Any]:
    msr = record.result.multi_seed_result
    m = msr.mean if msr is not None else record.result.metrics
    if m is None:
        return {}
    return {k: getattr(m, k, None) for k in _METRIC_COLS}


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.5f}"
    return str(v)


def generate_report(
    records: List[RunRecord],
    output_dir: Path,
    sort_metric: str = "nrmse_range",
) -> None:
    """Write report.md and report.csv into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for r in records:
        md = _get_metrics_dict(r)
        rows.append(
            {
                "reservoir": r.spec.model_type,
                "dataset": r.spec.dataset.name,
                "seed": r.spec.seed,
                "config_hash": r.result.config_hash,
                "timestamp": r.timestamp,
                "git_hash": r.git_hash or "",
                "status": r.result.status,
                **md,
            }
        )

    def _sort_key(row: Dict[str, Any]):
        v = row.get(sort_metric)
        return (v is None, v if v is not None else float("inf"))

    rows.sort(key=_sort_key)

    _write_markdown(rows, output_dir / "report.md", sort_metric, records)
    _write_csv(rows, output_dir / "report.csv")


def _write_markdown(
    rows: List[Dict[str, Any]],
    path: Path,
    sort_metric: str,
    records: List[RunRecord],
) -> None:
    lines: List[str] = []
    lines.append("# RC-Bench Report\n")

    timestamps = [r.timestamp for r in records if r.timestamp]
    if timestamps:
        lines.append(f"Generated: {timestamps[-1]}")
    lines.append(f"Records: {len(rows)}\n")

    hashes = sorted({r.git_hash for r in records if r.git_hash})
    if hashes:
        lines.append(f"Git: {', '.join(hashes)}\n")

    if not rows:
        path.write_text("\n".join(lines))
        return

    lines.append("## Summary\n")
    header = [
        "Reservoir", "Dataset", "Seed", "Status",
        "nrmse_range", "rmse", "prediction_horizon", "train_time (s)",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in rows:
        cells = [
            row.get("reservoir", "-"),
            row.get("dataset", "-"),
            str(row.get("seed", "-")),
            row.get("status", "-"),
            _fmt(row.get("nrmse_range")),
            _fmt(row.get("rmse")),
            _fmt(row.get("prediction_horizon")),
            _fmt(row.get("train_time")),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(f"*Sorted by `{sort_metric}` (ascending).*\n")

    # Per-dataset best (first entry after sort = best)
    by_dataset: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        ds = str(row.get("dataset", "?"))
        if ds not in by_dataset:
            by_dataset[ds] = row

    if len(by_dataset) > 1:
        lines.append("## Best per Dataset\n")
        for ds, row in by_dataset.items():
            lines.append(
                f"- **{ds}**: {row.get('reservoir', '?')} — "
                f"`{sort_metric}`={_fmt(row.get(sort_metric))}"
            )
        lines.append("")

    path.write_text("\n".join(lines))


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
