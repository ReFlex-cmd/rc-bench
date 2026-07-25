"""Evidence bundle: aggregation and traceability checks (EVID-001).

The bundle under ``reports/jmlc_2026/`` is what the defence cites, so every
number in it has to be re-derivable from the raw RunRecords sitting beside it.
This module does two jobs:

- ``build_rows`` / ``write_aggregates`` turn the raw per-cell RunRecords into
  the one table the README, slides and plots read from — nothing downstream
  recomputes a metric from predictions;
- ``validate_bundle`` re-checks the protocol invariants the bundle claims:
  one dataset pinned by digest, one shared evaluation context per horizon, the
  declared seeds actually evaluated, an equal HPO budget across reservoir
  models (DEC-004), and aggregates that agree with the raw records.

A check that cannot be expressed against the artifacts is deliberately absent
rather than approximated: the "test was never used for selection" property is
enforced structurally in the runners and their tests, not re-asserted here from
numbers that could not reveal a violation.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rc_bench.core.schema import MetricsResult, MetricsSummary
from rc_bench.reporting.run_record import RunRecord, load_run_record

# Metrics that must be present and finite in every published cell.
REQUIRED_FINITE_METRICS: Tuple[str, ...] = ("nrmse_std", "mae", "mase", "mae_skill")


@dataclass(frozen=True)
class CellRow:
    """One matrix cell as published in the aggregate table.

    ``*_sd`` columns are the across-seed sample standard deviations and are
    ``None`` for deterministic baselines, which are run exactly once (DEC-014)
    — an explicit absence, never a zero that would read as "no variance".
    """

    family: str
    model: str
    horizon: int
    # Режим протокола, в котором получена ячейка (fair | best_effort). Он
    # живёт в самой строке таблицы, а не только в имени каталога: имя файла
    # можно переименовать, строку — нет.
    mode: str
    deterministic: bool
    n_seeds: int
    evaluated_seeds: List[int]
    config_hash: str
    frozen_config_hash: Optional[str]
    selection_method: Optional[str]
    selection_metric: Optional[str]
    hpo_budget: Optional[int]
    n_test_targets: Optional[int]
    target_start_index: Optional[int]
    nrmse_std: float
    nrmse_std_sd: Optional[float]
    nrmse_range: float
    rmse: float
    mae: float
    mase: Optional[float]
    mase_sd: Optional[float]
    mae_skill: Optional[float]
    mae_skill_sd: Optional[float]
    val_nrmse_std: Optional[float]
    # Время обучения — первый пункт ресурсного профиля в описании проекта.
    # Оно измеряется в каждом прогоне, но до этой колонки не публиковалось,
    # то есть заявленный пункт нельзя было проверить по бандлу. Стоимость
    # ВЫВОДА живёт не здесь, а в profiles/summary.json: смешивать обучение с
    # инференсом в одной колонке — ровно та ошибка, ради которой профиль
    # выделен в отдельный проход.
    train_time_s: float
    train_time_s_sd: Optional[float]
    run_record: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_cells(runs_dir: str | Path) -> List[RunRecord]:
    """Load every RunRecord in *runs_dir*, sorted by file name."""
    runs_dir = Path(runs_dir)
    paths = sorted(runs_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no RunRecord JSON files under {runs_dir}")
    return [load_run_record(path) for path in paths]


def _metrics_of(record: RunRecord) -> Tuple[MetricsResult | MetricsSummary, Optional[MetricsSummary]]:
    """Return (point estimate, across-seed std) for a record.

    Multi-seed runs report the mean over seeds; single-run cells report their
    only measurement and have no spread to report.
    """
    msr = record.result.multi_seed_result
    if msr is not None:
        return msr.mean, msr.std
    if record.result.metrics is None:
        raise ValueError(f"record has neither metrics nor multi_seed_result: {record.result.config_hash}")
    return record.result.metrics, None


def _cell_identity(record: RunRecord) -> Tuple[str, str, int]:
    spec = record.spec
    family = record.result.model_family or spec.model_family
    return family, spec.model_type, spec.protocol.horizon


def build_rows(records: Sequence[RunRecord], runs_dir: str | Path) -> List[CellRow]:
    """Build the published table rows from raw records, sorted deterministically."""
    runs_dir = Path(runs_dir)
    rows: List[CellRow] = []
    for record in records:
        family, model, horizon = _cell_identity(record)
        mean, std = _metrics_of(record)
        result = record.result
        evaluation = result.evaluation
        selection = result.selection
        rows.append(
            CellRow(
                family=family,
                model=model,
                horizon=horizon,
                mode=record.spec.protocol.mode,
                deterministic=bool(result.deterministic),
                n_seeds=len(result.evaluated_seeds or []),
                evaluated_seeds=list(result.evaluated_seeds or []),
                config_hash=result.config_hash,
                frozen_config_hash=result.frozen_config_hash,
                selection_method=selection.method if selection is not None else None,
                selection_metric=selection.metric if selection is not None else None,
                hpo_budget=(
                    record.resolved_spec.protocol.hpo_budget
                    if record.resolved_spec.protocol.use_hpo
                    else None
                ),
                n_test_targets=evaluation.n_test_targets if evaluation else None,
                target_start_index=evaluation.target_start_index if evaluation else None,
                nrmse_std=mean.nrmse_std,
                nrmse_std_sd=std.nrmse_std if std is not None else None,
                nrmse_range=mean.nrmse_range,
                rmse=mean.rmse,
                mae=mean.mae,
                mase=mean.mase,
                mase_sd=std.mase if std is not None else None,
                mae_skill=mean.mae_skill,
                mae_skill_sd=std.mae_skill if std is not None else None,
                val_nrmse_std=mean.val_nrmse_std,
                train_time_s=mean.train_time,
                train_time_s_sd=std.train_time if std is not None else None,
                run_record=_bundle_relative(runs_dir, record, family, model, horizon),
            )
        )
    rows.sort(key=lambda row: (row.horizon, row.family, row.model))
    return rows


def _bundle_relative(runs_dir: Path, record: RunRecord, family: str, model: str, horizon: int) -> str:
    """Repo-relative POSIX path of the record's file (DEC-008: no absolute paths)."""
    del record
    path = runs_dir / f"{family}_{model}_h{horizon}.json"
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def aggregate_stem(mode: str) -> str:
    """Имя файла агрегата для режима.

    ``fair`` — headline-таблица бандла, поэтому сохраняет историческое имя
    ``matrix_table``; остальные режимы получают собственный файл. Одна таблица
    на два режима означала бы сравнение моделей, оценённых по разным правилам.
    """
    return "matrix_table" if mode == "fair" else f"matrix_table_{mode}"


def write_aggregates(rows: Sequence[CellRow], out_dir: str | Path) -> Dict[str, str]:
    """Write the aggregate table as JSON and CSV; return the written paths.

    Имя файла выводится из режима самих строк, а не задаётся параметром:
    иначе таблицу best-effort можно было бы записать под именем fair, и
    расхождение обнаружилось бы только на защите.
    """
    modes = {row.mode for row in rows}
    if len(modes) > 1:
        raise ValueError(
            f"table mixes protocol modes {sorted(modes)}; fair and best_effort "
            "results are published as separate tables"
        )
    mode = next(iter(modes)) if modes else "fair"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = aggregate_stem(mode)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    payload = [row.to_dict() for row in rows]
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    columns = [f.name for f in fields(CellRow)]
    with csv_path.open("w", newline="") as handle:
        # ``csv`` по умолчанию завершает строки CRLF независимо от платформы.
        # Для этого репозитория это ловушка: гейт (`verify.sh quick`) проверяет
        # `git diff --check`, который считает CR в конце строки лишним
        # пробелом, поэтому каждая регенерация бандла роняла бы гейт до
        # индексации файла.
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in payload:
            record = dict(row)
            record["evaluated_seeds"] = " ".join(str(s) for s in record["evaluated_seeds"])
            writer.writerow(record)

    return {"json": json_path.as_posix(), "csv": csv_path.as_posix()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def validate_records(
    records: Sequence[RunRecord],
    *,
    expected_cells: Optional[int] = None,
    fair_budget: Optional[int] = None,
) -> List[str]:
    """Return every protocol violation found in the raw records (empty = clean).

    ``fair_budget`` — бюджет HPO соответствующей fair-матрицы. Передаётся при
    проверке best-effort прогона: «лучшее усилие», оплаченное меньшим
    бюджетом, чем честное сравнение, — это не best-effort, а опечатка в
    конфиге, и опубликованный вывод «при индивидуальной настройке architecture
    X достигает Y» был бы получен в более бедных условиях, чем заявлено.
    """
    problems: List[str] = []
    if expected_cells is not None and len(records) != expected_cells:
        problems.append(f"expected {expected_cells} cells, found {len(records)}")

    identities = [_cell_identity(record) for record in records]
    duplicates = {ident for ident in identities if identities.count(ident) > 1}
    for ident in sorted(duplicates):
        problems.append(f"duplicate cell {ident[0]}/{ident[1]} h{ident[2]}")

    datasets = set()
    per_horizon_evaluation: Dict[int, List[Tuple[str, Any]]] = {}
    protocol_invariants = set()
    hpo_budgets = set()
    modes = set()

    for record, (family, model, horizon) in zip(records, identities):
        label = f"{family}/{model} h{horizon}"
        result = record.result

        if result.status != "completed":
            problems.append(f"{label}: status is {result.status!r}, not completed")
            continue

        if result.config_hash != record.resolved_spec.config_hash():
            problems.append(f"{label}: config_hash does not match its resolved spec")
        if result.frozen_config_hash != record.spec.config_hash():
            problems.append(f"{label}: frozen_config_hash does not match its frozen spec")

        dataset = record.spec.dataset
        datasets.add((dataset.name, dataset.length, dataset.raw_sha256))
        if dataset.raw_sha256 is None:
            problems.append(f"{label}: dataset.raw_sha256 is not pinned")

        protocol = record.spec.protocol
        modes.add(protocol.mode)
        protocol_invariants.add(
            (
                protocol.washout,
                protocol.train_frac,
                protocol.val_frac,
                protocol.forecasting_mode,
                protocol.seasonal_period,
            )
        )

        if result.evaluation is None:
            problems.append(f"{label}: no EvaluationContext recorded (DEC-018)")
        else:
            per_horizon_evaluation.setdefault(horizon, []).append(
                (label, result.evaluation.model_dump())
            )

        seeds = list(result.evaluated_seeds or [])
        if family == "baseline":
            if not result.deterministic:
                problems.append(f"{label}: baseline is not marked deterministic (DEC-014)")
            if seeds:
                problems.append(f"{label}: baseline reports evaluated seeds {seeds}")
            if protocol.use_hpo:
                problems.append(f"{label}: baseline must not run Optuna HPO")
        else:
            if result.deterministic:
                problems.append(f"{label}: reservoir is marked deterministic")
            if len(seeds) != protocol.n_seeds:
                problems.append(
                    f"{label}: declared n_seeds={protocol.n_seeds} but evaluated {len(seeds)}"
                )
            if protocol.use_hpo:
                hpo_budgets.add(protocol.hpo_budget)
                if (
                    protocol.mode == "best_effort"
                    and fair_budget is not None
                    and protocol.hpo_budget < fair_budget
                ):
                    problems.append(
                        f"{label}: best_effort budget {protocol.hpo_budget} is "
                        f"smaller than the fair budget {fair_budget}"
                    )
            elif protocol.mode == "best_effort":
                # Без этой ветки порог бюджета обходится целиком: ячейка без
                # HPO не попадает ни в одну проверку и публикуется как
                # «лучшее усилие», не сделав ни одного trial.
                problems.append(
                    f"{label}: best_effort reservoir cell did not run HPO, so its "
                    "budget cannot be compared with the fair one"
                )

        mean, _ = _metrics_of(record)
        for name in REQUIRED_FINITE_METRICS:
            if not _finite(getattr(mean, name, None)):
                problems.append(f"{label}: metric {name} is missing or non-finite")

    if len(datasets) > 1:
        problems.append(f"cells disagree on the dataset identity: {sorted(datasets)}")
    if len(protocol_invariants) > 1:
        problems.append(
            "cells disagree on shared protocol invariants "
            f"(washout/train_frac/val_frac/mode/seasonal_period): {sorted(protocol_invariants)}"
        )
    if len(modes) > 1:
        problems.append(
            f"bundle directory mixes protocol modes {sorted(modes)}; fair and "
            "best_effort matrices are published side by side, never merged"
        )
    # Равенство бюджетов — определение fair-режима (DEC-004). В best_effort
    # равные бюджеты означали бы, что режим не сделал того, ради чего он есть,
    # поэтому проверять их на равенство здесь было бы прямо неверно.
    if modes == {"fair"} and len(hpo_budgets) > 1:
        problems.append(
            f"reservoir models were given unequal HPO budgets {sorted(hpo_budgets)} (DEC-004)"
        )

    for horizon, entries in sorted(per_horizon_evaluation.items()):
        reference_label, reference = entries[0]
        for label, evaluation in entries[1:]:
            if evaluation != reference:
                problems.append(
                    f"h{horizon}: {label} was scored on a different evaluation context "
                    f"than {reference_label} (DEC-013)"
                )

    return problems


def validate_aggregates(rows: Sequence[CellRow], aggregates_path: str | Path) -> List[str]:
    """Check the published table still equals what the raw records say."""
    path = Path(aggregates_path)
    if not path.is_file():
        return [f"missing aggregate table: {path}"]

    published = json.loads(path.read_text())
    derived = [row.to_dict() for row in rows]
    if published == derived:
        return []
    return [
        f"{path.name} disagrees with the raw RunRecords; "
        "regenerate it with scripts/validate_evidence.py --write"
    ]


def discover_modes(bundle_dir: str | Path) -> List[str]:
    """Режимы, для которых в бандле есть прогоны, в порядке публикации."""
    bundle = Path(bundle_dir)
    return [
        mode
        for mode in ("fair", "best_effort")
        if (bundle / mode / "runs").is_dir()
    ]


def validate_bundle_modes(
    bundle_dir: str | Path,
    *,
    modes: Optional[Sequence[str]] = None,
    expected_cells: Optional[int] = 14,
    require_profiles: bool = True,
) -> List[str]:
    """Проверить каждый режим бандла отдельно, своим каталогом и агрегатом.

    Профиль ресурсов измеряется на fair-матрице, поэтому только она обязана
    его иметь: best-effort меняет гиперпараметры, а значит и стоимость шага,
    и переиспользовать под него fair-профиль было бы подлогом. Best-effort
    публикуется как контур качества, без стоимостных осей.

    Бюджет fair-матрицы передаётся в проверку best-effort: режим, оплаченный
    меньшим бюджетом, чем честное сравнение, своего названия не заслуживает.

    ``require_profiles=False`` снимает проверку профилей и с fair — это форма
    для незавершённого бандла, релизный гейт её не использует.
    """
    bundle = Path(bundle_dir)
    modes = list(modes) if modes is not None else discover_modes(bundle)
    problems: List[str] = []

    # Бюджет читается из fair-прогонов бандла всегда, когда они есть, а не
    # только когда fair попал в проверяемый список: `--modes best_effort` не
    # повод сравнивать best-effort не с чем.
    fair_budget: Optional[int] = None
    fair_runs = bundle / "fair" / "runs"
    if fair_runs.is_dir():
        budgets = {
            record.spec.protocol.hpo_budget
            for record in load_cells(fair_runs)
            if record.spec.protocol.use_hpo
            and record.spec.protocol.mode == "fair"
        }
        # Неравные бюджеты в fair — уже нарушение, о котором доложит
        # validate_records; здесь берётся минимум, чтобы не завышать порог.
        fair_budget = min(budgets) if budgets else None

    for mode in modes:
        mode_problems = validate_bundle(
            bundle,
            runs_subdir=f"{mode}/runs",
            expected_cells=expected_cells,
            require_profiles=(require_profiles and mode == "fair"),
            fair_budget=fair_budget if mode != "fair" else None,
            expected_mode=mode,
        )
        problems.extend(f"[{mode}] {problem}" for problem in mode_problems)

    return problems


def validate_bundle(
    bundle_dir: str | Path,
    *,
    runs_subdir: str = "fair/runs",
    expected_cells: Optional[int] = 14,
    require_profiles: bool = True,
    fair_budget: Optional[int] = None,
    expected_mode: Optional[str] = None,
) -> List[str]:
    """Validate a full evidence bundle; return the list of problems found.

    ``expected_mode`` сверяет режим, объявленный записями, с режимом, под
    именем которого они лежат. Это два независимых утверждения об одном
    факте, и раннер пишет в каталог, выбранный оператором руками, поэтому
    расхождение достижимо обычной опечаткой.
    """
    bundle = Path(bundle_dir)
    problems: List[str] = []

    manifest_path = bundle / "dataset_manifest.json"
    manifest_digest: Optional[str] = None
    if not manifest_path.is_file():
        problems.append(f"missing dataset manifest: {manifest_path}")
    else:
        try:
            manifest_digest = json.loads(manifest_path.read_text())["raw_file"]["sha256"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            problems.append(f"unreadable dataset manifest {manifest_path}: {exc}")

    if not (bundle / "hardware_profile.json").is_file():
        problems.append(f"missing hardware profile: {bundle / 'hardware_profile.json'}")

    runs_dir = bundle / runs_subdir
    if not runs_dir.is_dir():
        problems.append(f"missing matrix runs directory: {runs_dir}")
        return problems

    records = load_cells(runs_dir)
    if expected_mode is not None:
        declared = sorted({record.spec.protocol.mode for record in records})
        if declared and declared != [expected_mode]:
            problems.append(
                f"{runs_subdir}: records declare mode {declared} but live under "
                f"the {expected_mode!r} contour"
            )

    problems.extend(
        validate_records(
            records, expected_cells=expected_cells, fair_budget=fair_budget
        )
    )

    if manifest_digest is not None:
        for record in records:
            digest = record.spec.dataset.raw_sha256
            if digest is not None and digest != manifest_digest:
                family, model, horizon = _cell_identity(record)
                problems.append(
                    f"{family}/{model} h{horizon}: raw_sha256 {digest[:12]}… does not match "
                    f"the bundle manifest {manifest_digest[:12]}…"
                )

    rows = build_rows(records, runs_dir)
    stem = aggregate_stem(rows[0].mode) if rows else "matrix_table"
    problems.extend(validate_aggregates(rows, bundle / "aggregates" / f"{stem}.json"))

    if require_profiles:
        problems.extend(_validate_profiles(bundle, rows))

    return problems


def _validate_profiles(bundle: Path, rows: Sequence[CellRow]) -> List[str]:
    """Every published cell needs a resource profile it can be plotted against."""
    problems: List[str] = []
    summary_path = bundle / "profiles" / "summary.json"
    if not summary_path.is_file():
        return [f"missing resource-profile summary: {summary_path}"]

    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"unreadable resource-profile summary {summary_path}: {exc}"]

    profiled = {
        (entry.get("family"), entry.get("model"), entry.get("horizon")): entry
        for entry in summary
    }
    for row in rows:
        entry = profiled.get((row.family, row.model, row.horizon))
        label = f"{row.family}/{row.model} h{row.horizon}"
        if entry is None:
            problems.append(f"{label}: no resource profile in profiles/summary.json")
            continue
        if entry.get("status") != "completed":
            problems.append(f"{label}: resource profile status is {entry.get('status')!r}")
        if entry.get("config_hash") != row.config_hash:
            problems.append(
                f"{label}: profile was measured on config {entry.get('config_hash')}, "
                f"but the published result is {row.config_hash}"
            )
    return problems
