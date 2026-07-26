"""Выбор архитектуры под ограничения устройства (SELECT-001).

Отбор идёт по опубликованным артефактам бандла, а не по свежему прогону:
пользователь должен получать тот же ответ, что и проверяющий, читающий
таблицу руками. Поэтому источники названы прямо в отчёте, а профиль,
измеренный на другой конфигурации, отвергается — как и в Pareto-построителе.

Ограничение, для которого нет измерения (энергия без счётчика), не
превращается в отказ: оно объявляется неприменимым и называется в отчёте.
Молча отбросить все модели по ограничению, которое нечем проверить, — это
неверный ответ, выданный с уверенным видом.

Победитель — допустимый кандидат с минимальным NRMSE_std. Это ответ на один
конкретный вопрос («что взять при таком бюджете»), поэтому рядом печатается
Pareto-фронт: по нему видно, какой размен качества и стоимости был бы при
другом бюджете, и что победитель — не единственная разумная точка.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rc_bench.reporting.pareto import (
    COST_AXES,
    MissingCostAxis,
    ParetoPoint,
    load_points,
    pareto_front,
)

AGGREGATE_REL = "aggregates/matrix_table.json"
PROFILES_REL = "profiles/summary.json"


@dataclass(frozen=True)
class DeviceConstraints:
    """Бюджет целевого устройства. ``None`` — ограничения нет.

    Все пороги — «не более»: модель, попавшая ровно в бюджет, устройство
    выдерживает, и отбрасывать её было бы произволом.
    """

    max_p50_us: Optional[float] = None
    max_state_bytes: Optional[int] = None
    max_model_bytes: Optional[int] = None
    max_energy_mj: Optional[float] = None

    def declared(self) -> List[str]:
        """Заявленные ограничения в порядке объявления, для отчёта."""
        rows = [
            ("задержка шага p50", self.max_p50_us, "мкс"),
            ("рабочее состояние", self.max_state_bytes, "байт"),
            ("сериализованная модель", self.max_model_bytes, "байт"),
            ("энергия одного вывода", self.max_energy_mj, "мДж"),
        ]
        return [f"{name} ≤ {value} {unit}" for name, value, unit in rows if value is not None]


@dataclass
class Candidate:
    model: str
    family: str
    horizon: int
    nrmse_std: float
    p50_us: float
    state_bytes: int
    model_bytes: int
    energy_mj: Optional[float]
    feasible: bool = True
    violations: List[str] = field(default_factory=list)


@dataclass
class SelectionReport:
    bundle_dir: str
    horizon: int
    constraints: DeviceConstraints
    candidates: List[Candidate]
    winner: Optional[Candidate]
    frontiers: Dict[str, List[ParetoPoint]]
    energy_status: str
    #: Ограничения, объявленные пользователем, но неприменимые по этому бандлу.
    inapplicable: List[str] = field(default_factory=list)


def _load_cells(bundle: Path, horizon: int) -> List[Dict]:
    """Строки таблицы, соединённые со своими профилями по config_hash."""
    rows = json.loads((bundle / AGGREGATE_REL).read_text())
    profiles = json.loads((bundle / PROFILES_REL).read_text())
    by_cell = {
        (entry.get("family"), entry.get("model"), entry.get("horizon")): entry
        for entry in profiles
    }

    joined: List[Dict] = []
    for row in rows:
        if row["horizon"] != horizon:
            continue
        key = (row["family"], row["model"], row["horizon"])
        profile = by_cell.get(key)
        if profile is None:
            raise ValueError(f"no resource profile for {key[0]}/{key[1]} h{key[2]}")
        if profile.get("config_hash") != row["config_hash"]:
            raise ValueError(
                f"{key[0]}/{key[1]} h{key[2]}: profile config {profile.get('config_hash')} "
                f"does not match the published result {row['config_hash']}"
            )
        joined.append({"row": row, "profile": profile})
    if not joined:
        raise ValueError(f"no published cells for horizon {horizon} in {bundle}")
    return joined


def _energy_status(cells: List[Dict]) -> str:
    """``measured`` только если энергия есть у каждой ячейки горизонта.

    Смешанный случай — это тот же неполный контур, что отвергает Pareto:
    сравнивать модели по оси, которой у части из них нет, значит сравнивать
    разные множества.
    """
    values = [cell["profile"].get("net_energy_per_inference_mj") for cell in cells]
    return "measured" if values and all(v is not None for v in values) else "unavailable"


def _violations(candidate: Candidate, constraints: DeviceConstraints, *, energy: bool) -> List[str]:
    problems: List[str] = []
    if constraints.max_p50_us is not None and candidate.p50_us > constraints.max_p50_us:
        problems.append(
            f"p50 {candidate.p50_us:.2f} мкс > {constraints.max_p50_us} мкс"
        )
    if constraints.max_state_bytes is not None and candidate.state_bytes > constraints.max_state_bytes:
        problems.append(
            f"состояние {candidate.state_bytes} B > {constraints.max_state_bytes} B"
        )
    if constraints.max_model_bytes is not None and candidate.model_bytes > constraints.max_model_bytes:
        problems.append(
            f"модель {candidate.model_bytes} B > {constraints.max_model_bytes} B"
        )
    if (
        energy
        and constraints.max_energy_mj is not None
        and candidate.energy_mj is not None
        and candidate.energy_mj > constraints.max_energy_mj
    ):
        problems.append(
            f"energy {candidate.energy_mj:.4f} мДж > {constraints.max_energy_mj} мДж"
        )
    return problems


def _frontiers(bundle: Path, horizon: int) -> Dict[str, List[ParetoPoint]]:
    """Pareto-фронт по каждой доступной оси стоимости, для этого горизонта."""
    frontiers: Dict[str, List[ParetoPoint]] = {}
    for cost in COST_AXES:
        try:
            points = [p for p in load_points(bundle, cost) if p.horizon == horizon]
        except MissingCostAxis:
            continue
        if points:
            frontiers[cost] = pareto_front(points)
    return frontiers


def select(
    bundle_dir: str | Path,
    *,
    horizon: int,
    constraints: DeviceConstraints,
) -> SelectionReport:
    """Выбрать лучшую по качеству модель, укладывающуюся в бюджет устройства."""
    bundle = Path(bundle_dir)
    cells = _load_cells(bundle, horizon)
    energy_status = _energy_status(cells)

    inapplicable: List[str] = []
    if constraints.max_energy_mj is not None and energy_status != "measured":
        inapplicable.append(
            f"энергия одного вывода ≤ {constraints.max_energy_mj} мДж "
            "(энергия недоступна: счётчик не был доступен при профилировании)"
        )

    candidates: List[Candidate] = []
    for cell in cells:
        row, profile = cell["row"], cell["profile"]
        candidate = Candidate(
            model=row["model"],
            family=row["family"],
            horizon=row["horizon"],
            nrmse_std=float(row["nrmse_std"]),
            p50_us=float(profile["deployable_p50_ns"]) / 1e3,
            state_bytes=int(profile["working_state_bytes"]),
            model_bytes=int(profile["serialized_model_bytes"]),
            energy_mj=profile.get("net_energy_per_inference_mj"),
        )
        candidate.violations = _violations(
            candidate, constraints, energy=(energy_status == "measured")
        )
        candidate.feasible = not candidate.violations
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.nrmse_std)
    feasible = [c for c in candidates if c.feasible]
    winner = feasible[0] if feasible else None

    return SelectionReport(
        bundle_dir=bundle.as_posix(),
        horizon=horizon,
        constraints=constraints,
        candidates=candidates,
        winner=winner,
        frontiers=_frontiers(bundle, horizon),
        energy_status=energy_status,
        inapplicable=inapplicable,
    )


def _energy_cell(candidate: Candidate) -> str:
    return "—" if candidate.energy_mj is None else f"{candidate.energy_mj:.4f}"


def render_markdown(report: SelectionReport) -> str:
    """Отчёт выбора: ограничения, все кандидаты с причинами, победитель, фронт.

    Отвергнутые кандидаты остаются в таблице: читатель должен видеть, что́ было
    рассмотрено и почему отброшено, иначе отчёт неотличим от списка того, что
    и так подошло.
    """
    lines: List[str] = [
        f"# Выбор архитектуры под ограничения устройства (горизонт {report.horizon})",
        "",
        f"Источники: `{report.bundle_dir}/{AGGREGATE_REL}`, "
        f"`{report.bundle_dir}/{PROFILES_REL}`.",
        "",
        "## Ограничения",
        "",
    ]

    declared = report.constraints.declared()
    lines.extend([f"- {item}" for item in declared] if declared else ["- не заданы"])

    if report.inapplicable:
        lines += ["", "Неприменимо по этому бандлу:"]
        lines += [f"- {item}" for item in report.inapplicable]

    lines += [
        "",
        "## Кандидаты",
        "",
        "| модель | семейство | NRMSE_std | p50, мкс | состояние, B | модель, B | энергия, мДж | допустим | причина |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for candidate in report.candidates:
        lines.append(
            f"| {candidate.model} | {candidate.family} | {candidate.nrmse_std:.4f} | "
            f"{candidate.p50_us:.2f} | {candidate.state_bytes} | {candidate.model_bytes} | "
            f"{_energy_cell(candidate)} | {'да' if candidate.feasible else 'нет'} | "
            f"{'; '.join(candidate.violations) or '—'} |"
        )

    lines += ["", "## Результат", ""]
    if report.winner is None:
        lines.append(
            "**Нет допустимых кандидатов**: ни одна опубликованная модель не "
            "укладывается в заданный бюджет. Ослабьте ограничение или снимите "
            "профиль на целевом устройстве."
        )
    else:
        lines.append(
            f"**{report.winner.model}** (NRMSE_std {report.winner.nrmse_std:.4f}, "
            f"p50 {report.winner.p50_us:.2f} мкс, состояние "
            f"{report.winner.state_bytes} B) — лучшая по качеству среди "
            "укладывающихся в бюджет."
        )

    lines += [
        "",
        "## Pareto-фронт",
        "",
        "Победитель — ответ на один заданный бюджет; фронт показывает, какой "
        "размен качества и стоимости доступен при другом.",
        "",
    ]
    if report.frontiers:
        for cost, front in report.frontiers.items():
            names = ", ".join(f"{p.model} ({p.cost:.4g})" for p in front)
            lines.append(f"- **{cost}**: {names}")
    else:
        lines.append("- нет осей стоимости, измеренных для всех ячеек горизонта")

    lines += [
        "",
        f"Статус энергетического измерения: `{report.energy_status}`.",
        "",
    ]
    return "\n".join(lines)
