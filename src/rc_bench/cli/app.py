"""
rcbench — RC-Bench command-line interface.

Commands
--------
list-datasets     Print available datasets.
list-reservoirs   Print available reservoir types.
validate-spec     Validate a JSON/YAML ExperimentSpec file.
run               Run a single experiment from a spec file.
aggregate         Summarise multiple run-record JSON files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from rc_bench.reporting.selection import (
    DeviceConstraints,
    render_markdown,
    select,
)

app = typer.Typer(
    name="rcbench",
    help="RC-Bench: reproducible reservoir computing benchmark suite.",
    add_completion=False,
)
console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_spec_file(path: Path) -> dict:
    """Load JSON or YAML file and return a plain dict."""
    raw = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(raw)
    return json.loads(raw)


def _metrics_table(title: str) -> Table:
    t = Table(title=title, show_header=True, header_style="bold cyan")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    return t


def _print_result(result_spec) -> None:
    """Print a rich metrics table from a ResultSpec (single or multi-seed)."""
    msr = result_spec.multi_seed_result

    if msr is not None:
        # Multi-seed: show mean ± std
        t = _metrics_table(f"Metrics  (mean ± std,  n={msr.n_seeds})")
        m, s = msr.mean, msr.std
        for field in ("rmse", "nrmse_range", "nrmse_std", "nrmse_var",
                      "mae", "mse", "val_nrmse_range"):
            t.add_row(field, f"{getattr(m, field):.6f} ± {getattr(s, field):.6f}")
        # Baseline-relative metrics only exist when a seasonal period was given
        # (DEC-013); a row of dashes would read as "measured, and it is nothing".
        for field in ("mase", "mae_skill"):
            if getattr(m, field, None) is not None:
                t.add_row(field, f"{getattr(m, field):.6f} ± {getattr(s, field):.6f}")
        t.add_row("prediction_horizon",
                  f"{m.prediction_horizon:.1f} ± {s.prediction_horizon:.1f}")
        t.add_row("train_time (s)",
                  f"{m.train_time:.4f} ± {s.train_time:.4f}")
        t.add_row("inference_latency (s)",
                  f"{m.inference_latency:.4f} ± {s.inference_latency:.4f}")
        t.add_row("peak_memory (KB)",
                  f"{m.peak_memory / 1024:.0f} ± {s.peak_memory / 1024:.0f}")
    else:
        metrics = result_spec.metrics
        t = _metrics_table("Metrics")
        t.add_row("rmse",               f"{metrics.rmse:.6f}")
        t.add_row("nrmse_range",        f"{metrics.nrmse_range:.6f}")
        t.add_row("nrmse_std",          f"{metrics.nrmse_std:.6f}")
        t.add_row("nrmse_var",          f"{metrics.nrmse_var:.6f}")
        t.add_row("mae",                f"{metrics.mae:.6f}")
        t.add_row("mse",                f"{metrics.mse:.6f}")
        if metrics.mase is not None:
            t.add_row("mase",           f"{metrics.mase:.6f}")
        if metrics.mae_skill is not None:
            t.add_row("mae_skill",      f"{metrics.mae_skill:.6f}")
        t.add_row("prediction_horizon", str(metrics.prediction_horizon))
        t.add_row("val_nrmse_range",    f"{metrics.val_nrmse_range:.6f}")
        t.add_row("train_time (s)",     f"{metrics.train_time:.4f}")
        t.add_row("inference_latency (s)", f"{metrics.inference_latency:.4f}")
        t.add_row("peak_memory (KB)",   f"{metrics.peak_memory // 1024}")

    if result_spec.hpo_best_params:
        console.print("[cyan]HPO best params:[/cyan]",
                      result_spec.hpo_best_params)
    console.print(t)


# ---------------------------------------------------------------------------
# list-datasets
# ---------------------------------------------------------------------------

@app.command("list-datasets")
def list_datasets() -> None:
    """Print all available datasets."""
    from rc_bench.core.data_provider import DATASET_CATALOG

    t = Table(title="Available Datasets", show_header=True, header_style="bold cyan")
    t.add_column("Name", style="bold")
    t.add_column("Description")
    t.add_column("Default length", justify="right")
    t.add_column("Input dim", justify="right")
    t.add_column("Output dim", justify="right")

    for name, info in DATASET_CATALOG.items():
        t.add_row(
            name,
            info["description"],
            str(info["default_length"]),
            str(info["input_dim"]),
            str(info["output_dim"]),
        )
    console.print(t)


# ---------------------------------------------------------------------------
# list-reservoirs
# ---------------------------------------------------------------------------

@app.command("list-reservoirs")
def list_reservoirs() -> None:
    """Print all registered reservoir types."""
    from rc_bench.core.reservoirs.registry import REGISTRY

    t = Table(title="Available Reservoir Types", show_header=True, header_style="bold cyan")
    t.add_column("Type", style="bold")
    t.add_column("Class")
    t.add_column("Default scaler")

    for name, cls in sorted(REGISTRY.items()):
        t.add_row(name, cls.__name__, cls.DEFAULT_SCALER)
    console.print(t)


# ---------------------------------------------------------------------------
# validate-spec
# ---------------------------------------------------------------------------

@app.command("validate-spec")
def validate_spec(
    spec_file: Path = typer.Argument(..., help="Path to JSON or YAML ExperimentSpec file."),
) -> None:
    """Validate a JSON or YAML ExperimentSpec file."""
    from rc_bench.core.schema import ExperimentSpec

    if not spec_file.exists():
        console.print(f"[red]File not found:[/red] {spec_file}")
        raise typer.Exit(1)

    try:
        raw = _load_spec_file(spec_file)
        spec = ExperimentSpec.model_validate(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        console.print(f"[red]Parse error:[/red] {exc}")
        raise typer.Exit(1)
    except ValidationError as exc:
        console.print(f"[red]Validation failed[/red] ({exc.error_count()} error(s)):")
        for err in exc.errors():
            loc = " → ".join(str(l) for l in err["loc"])
            console.print(f"  [yellow]{loc}[/yellow]: {err['msg']}")
        raise typer.Exit(1)

    console.print(f"[green]✓ Valid[/green]  config_hash={spec.config_hash()}")
    console.print_json(spec.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command("run")
def run_cmd(
    spec_file: Path = typer.Argument(..., help="Path to JSON or YAML ExperimentSpec file."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Write RunRecord (spec + metrics + metadata) to this JSON file.",
    ),
    artifacts: Optional[Path] = typer.Option(
        None, "--artifacts", "-a",
        help="Directory to save prediction artifacts (.npz).",
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s",
        help="Override the seed in the spec.",
    ),
) -> None:
    """Run an experiment (with optional HPO and multi-seed) from a spec file."""
    from rc_bench.core.data_provider import get_data_for_experiment
    from rc_bench.core.schema import ExperimentSpec
    from rc_bench.reporting.run_record import RunRecord, save_run_record
    from rc_bench.runners.pipeline import run_pipeline

    if not spec_file.exists():
        console.print(f"[red]File not found:[/red] {spec_file}")
        raise typer.Exit(1)

    try:
        raw = _load_spec_file(spec_file)
        spec = ExperimentSpec.model_validate(raw)
    except (json.JSONDecodeError, yaml.YAMLError, ValidationError) as exc:
        console.print(f"[red]Spec error:[/red] {exc}")
        raise typer.Exit(1)

    if seed is not None:
        spec = spec.model_copy(update={"seed": seed})

    hpo_tag = " + HPO" if spec.protocol.use_hpo else ""
    seed_tag = f" × {spec.protocol.n_seeds} seeds" if spec.protocol.n_seeds > 1 else ""
    console.rule(
        f"[bold cyan]rcbench run  —  {spec.reservoir.type} / {spec.dataset.name}"
        f"{hpo_tag}{seed_tag}"
    )

    try:
        data = get_data_for_experiment(
            dataset_name=spec.dataset.name,
            length=spec.dataset.length,
            train_frac=spec.protocol.train_frac,
            val_frac=spec.protocol.val_frac,
            seed=spec.dataset.seed,
        )
        result_spec = run_pipeline(
            data,
            spec,
            artifact_dir=artifacts,
            save_predictions=artifacts is not None,
        )
    except Exception as exc:
        console.print(f"[red]Run failed:[/red] {exc}")
        raise typer.Exit(1)

    _print_result(result_spec)

    if output is not None:
        record = RunRecord.make(spec, result_spec)
        save_run_record(record, output)
        console.print(f"[green]Saved →[/green] {output}")


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

@app.command("aggregate")
def aggregate(
    results_dir: Path = typer.Argument(
        ..., help="Directory containing run-record JSON files (*.json)."
    ),
    sort_by: str = typer.Option(
        "nrmse_range", "--sort-by", help="Metric column to sort by (ascending)."
    ),
    top: int = typer.Option(0, "--top", help="Show only the N best rows (0 = all)."),
) -> None:
    """Aggregate run records from a directory and print a comparison table."""
    if not results_dir.is_dir():
        console.print(f"[red]Not a directory:[/red] {results_dir}")
        raise typer.Exit(1)

    files = sorted(results_dir.glob("*.json"))
    if not files:
        console.print(f"[yellow]No JSON files found in {results_dir}[/yellow]")
        raise typer.Exit(0)

    METRIC_COLS = [
        "nrmse_range", "nrmse_std", "nrmse_var", "rmse", "mae",
        "prediction_horizon", "train_time", "inference_latency",
    ]

    rows = []
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError:
            console.print(f"[yellow]Skipping (invalid JSON):[/yellow] {f.name}")
            continue

        spec_d = rec.get("spec", {})
        metrics_d = (rec.get("result") or {}).get("metrics") or {}
        status = (rec.get("result") or {}).get("status", "?")

        rows.append({
            "file": f.stem,
            "reservoir": (spec_d.get("reservoir") or {}).get("type", "?"),
            "dataset": (spec_d.get("dataset") or {}).get("name", "?"),
            "seed": spec_d.get("seed", "?"),
            "status": status,
            **{col: metrics_d.get(col) for col in METRIC_COLS},
        })

    if not rows:
        console.print("[yellow]No valid run records found.[/yellow]")
        raise typer.Exit(0)

    # Sort
    def _sort_key(r: dict):
        v = r.get(sort_by)
        return (v is None, v if v is not None else 0)

    rows.sort(key=_sort_key)
    if top > 0:
        rows = rows[:top]

    t = Table(
        title=f"Aggregate Results  (sorted by {sort_by})",
        show_header=True,
        header_style="bold cyan",
    )
    for col in ["file", "reservoir", "dataset", "seed", "status"] + METRIC_COLS:
        justify = "right" if col not in {"file", "reservoir", "dataset", "status"} else "left"
        t.add_column(col, justify=justify)

    for r in rows:
        def _fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.5f}"
            return str(v)

        t.add_row(*[_fmt(r[col]) for col in ["file", "reservoir", "dataset", "seed", "status"] + METRIC_COLS])

    console.print(t)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@app.command("report")
def report_cmd(
    results_dir: Path = typer.Argument(
        ..., help="Directory containing run-record JSON files (*.json)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir",
        help="Where to write report.md, report.csv, comparison.png. Defaults to results_dir.",
    ),
    metric: str = typer.Option(
        "nrmse_range", "--metric",
        help="Metric used for sorting and the comparison bar chart.",
    ),
    no_plots: bool = typer.Option(
        False, "--no-plots", help="Skip matplotlib plot generation."
    ),
) -> None:
    """Generate a Markdown/CSV report and comparison plot from run-record JSON files."""
    from rc_bench.reporting.report import generate_report
    from rc_bench.reporting.run_record import load_run_record

    if not results_dir.is_dir():
        console.print(f"[red]Not a directory:[/red] {results_dir}")
        raise typer.Exit(1)

    files = sorted(results_dir.glob("*.json"))
    if not files:
        console.print(f"[yellow]No JSON files found in {results_dir}[/yellow]")
        raise typer.Exit(0)

    records = []
    for f in files:
        try:
            records.append(load_run_record(f))
        except Exception as exc:
            console.print(f"[yellow]Skipping {f.name}:[/yellow] {exc}")

    if not records:
        console.print("[yellow]No valid run records found.[/yellow]")
        raise typer.Exit(0)

    out = output_dir or results_dir
    generate_report(records, out, sort_metric=metric)
    console.print(f"[green]report.md[/green] → {out / 'report.md'}")
    console.print(f"[green]report.csv[/green] → {out / 'report.csv'}")

    if not no_plots:
        try:
            from rc_bench.reporting.plots import plot_metric_bar
            bar_path = out / f"comparison_{metric}.png"
            plot_metric_bar(records, metric=metric, output_path=bar_path)
            console.print(f"[green]{bar_path.name}[/green] → {bar_path}")
        except Exception as exc:
            console.print(f"[yellow]Plot skipped:[/yellow] {exc}")


# ---------------------------------------------------------------------------
# eda
# ---------------------------------------------------------------------------

@app.command("eda")
def eda_cmd(
    raw_path: Path = typer.Argument(
        ...,
        help="Verified UCI household_power_consumption.txt path.",
    ),
    manifest: Path = typer.Option(
        Path("configs/jmlc/dataset_manifest.json"),
        "--manifest",
        help="Pinned dataset manifest used for local size/SHA-256 verification.",
    ),
    output_dir: Path = typer.Option(
        Path("reports/jmlc_2026"),
        "--output-dir",
        help="Evidence root for the fixed EDA Markdown, JSON and plot files.",
    ),
) -> None:
    """Generate deterministic observed-only EDA for the pinned 12k window."""
    from rc_bench.data.download import (
        DatasetDownloadError,
        DatasetManifest,
        download_dataset,
    )
    from rc_bench.data.jmlc import (
        JMLCDataError,
        load_uci_household_power_series,
    )
    from rc_bench.reporting.eda import (
        EDAInvariantError,
        generate_eda_report,
    )

    if not raw_path.is_file():
        console.print(f"[red]Raw data file not found:[/red] {raw_path}")
        raise typer.Exit(1)

    try:
        pinned = DatasetManifest.load(manifest)
        if raw_path.name != pinned.raw_filename:
            raise ValueError(
                f"raw filename {raw_path.name!r} does not match manifest "
                f"filename {pinned.raw_filename!r}"
            )

        verified_path = download_dataset(manifest, raw_path.parent)
        if verified_path.resolve() != raw_path.resolve():
            raise ValueError(
                "verified manifest path does not match the requested raw path"
            )

        series = load_uci_household_power_series(verified_path)
        paths = generate_eda_report(
            series,
            output_dir,
            raw_sha256=pinned.raw_sha256,
        )
    except (
        DatasetDownloadError,
        EDAInvariantError,
        JMLCDataError,
        OSError,
        ValueError,
    ) as exc:
        console.print(f"[red]EDA failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print("[green]EDA complete[/green]")
    for path in paths:
        console.print(f"  {path.relative_to(output_dir).as_posix()}")


@app.command("select")
def select_command(
    bundle: Path = typer.Option(..., "--bundle", help="Evidence bundle directory."),
    horizon: int = typer.Option(1, "--horizon", help="Forecast horizon to select for."),
    max_latency_us: Optional[float] = typer.Option(
        None, "--max-latency-us", help="Budget for the deployable p50 step latency."
    ),
    max_state_bytes: Optional[int] = typer.Option(
        None, "--max-state-bytes", help="Budget for the working state."
    ),
    max_model_bytes: Optional[int] = typer.Option(
        None, "--max-model-bytes", help="Budget for the serialized model."
    ),
    max_energy_mj: Optional[float] = typer.Option(
        None,
        "--max-energy-mj",
        help=(
            "Budget for the energy of one inference; ignored (and said so in "
            "the report) when the bundle carries no energy measurement."
        ),
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Also write the Markdown report to this path."
    ),
) -> None:
    """Find the Pareto-optimal model that fits a device's budget."""
    constraints = DeviceConstraints(
        max_p50_us=max_latency_us,
        max_state_bytes=max_state_bytes,
        max_model_bytes=max_model_bytes,
        max_energy_mj=max_energy_mj,
    )
    try:
        report = select(bundle, horizon=horizon, constraints=constraints)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Selection failed:[/red] {exc}")
        raise typer.Exit(1)

    markdown = render_markdown(report)
    # print, а не console.print: rich разметил бы Markdown-таблицу по-своему,
    # и вывод перестал бы совпадать с файлом, который пишет --output.
    print(markdown)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
