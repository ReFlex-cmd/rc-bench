"""Измерение энергии вывода через Intel RAPL (ENERGY-002).

Протокол (docs/PROJECT_CONTRACT.md, «Energy»):

- источник — powercap-интерфейс ядра, ``/sys/class/powercap/intel-rapl:*``;
  берутся домены верхнего уровня (package), а не их поддомены, чтобы не
  сложить одну и ту же энергию дважды;
- окно измерения не короче ``min_duration_s`` (по умолчанию 2 с) И не меньше
  ``min_steps`` шагов: разрешение счётчика ~61 мкДж и период обновления ~1 мс
  сопоставимы с длительностью одного шага (десятки микросекунд), поэтому
  измерять один вызов бессмысленно;
- переполнение счётчика обрабатывается по ``max_energy_range_uj``;
- отдельно измеряется энергия простоя той же длительности, и публикуются
  обе величины: полная (пакет целиком) и net (за вычетом простоя). Без
  вычитания простоя число говорило бы в основном о статическом потреблении
  процессора, а не о модели;
- при отсутствии или недоступности счётчика возвращается
  ``status="unavailable"`` с причиной. Прокси-метрики сюда не подставляются
  никогда: energy и activity — разные блоки (DEC-007).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

RAPL_ROOT = Path("/sys/class/powercap")


@dataclass(frozen=True)
class RaplDomain:
    name: str
    energy_path: Path
    max_range_uj: int


def discover_rapl_domains(root: Path = RAPL_ROOT) -> List[RaplDomain]:
    """Домены верхнего уровня (``intel-rapl:N``), без поддоменов ``:N:M``."""
    root = Path(root)
    if not root.is_dir():
        return []
    domains: List[RaplDomain] = []
    for entry in sorted(root.glob("intel-rapl:*")):
        if entry.name.count(":") != 1:
            continue
        name_file, energy_file = entry / "name", entry / "energy_uj"
        range_file = entry / "max_energy_range_uj"
        if not (name_file.is_file() and energy_file.is_file()):
            continue
        try:
            max_range = int(range_file.read_text().strip())
        except (OSError, ValueError):
            max_range = 2**32 - 1
        domains.append(
            RaplDomain(name_file.read_text().strip(), energy_file, max_range)
        )
    return domains


def read_domain_uj(domain: RaplDomain) -> int:
    return int(domain.energy_path.read_text().strip())


def counter_delta_uj(*, start_uj: int, end_uj: int, domain: RaplDomain) -> int:
    """Разность с учётом кольцевого переполнения счётчика."""
    if end_uj >= start_uj:
        return end_uj - start_uj
    return (domain.max_range_uj - start_uj) + end_uj + 1


def energy_backend_status(root: Path = RAPL_ROOT) -> Dict[str, Any]:
    domains = discover_rapl_domains(root)
    if not domains:
        return {
            "status": "unavailable",
            "backend": None,
            "reason": f"no powercap intel-rapl domains under {root}",
        }
    try:
        for domain in domains:
            read_domain_uj(domain)
    except OSError as exc:
        return {
            "status": "unavailable",
            "backend": None,
            "reason": (
                f"intel-rapl counter {exc.filename} is not readable by this user "
                "(root-only since CVE-2020-8694); see the plan's prerequisite"
            ),
        }
    return {
        "status": "available",
        "backend": "intel_rapl",
        "domains": [d.name for d in domains],
    }


def _sum_uj(domains: List[RaplDomain]) -> List[int]:
    return [read_domain_uj(d) for d in domains]


def _elapsed_uj(domains: List[RaplDomain], start: List[int], end: List[int]) -> float:
    return float(
        sum(
            counter_delta_uj(start_uj=s, end_uj=e, domain=d)
            for d, s, e in zip(domains, start, end)
        )
    )


def measure_energy(
    step_fn: Callable[[], Any],
    *,
    p50_ns: float,
    min_duration_s: float = 2.0,
    min_steps: int = 1000,
    root: Path = RAPL_ROOT,
) -> Dict[str, Any]:
    """Энергия одного шага вывода ``step_fn``.

    ``p50_ns`` — уже измеренная медианная задержка шага: по ней выбирается
    число шагов, чтобы окно измерения было длиннее периода обновления
    счётчика. Она же используется для energy-delay product.
    """
    status = energy_backend_status(root)
    if status["status"] != "available":
        return status

    domains = discover_rapl_domains(root)
    n_steps = max(int(min_steps), int((min_duration_s * 1e9) // max(p50_ns, 1.0)))

    # Простой той же длительности — базовая линия статического потребления.
    idle_start = _sum_uj(domains)
    t0 = time.perf_counter_ns()
    time.sleep(min_duration_s)
    idle_duration_s = (time.perf_counter_ns() - t0) / 1e9
    idle_uj = _elapsed_uj(domains, idle_start, _sum_uj(domains))

    start = _sum_uj(domains)
    t0 = time.perf_counter_ns()
    for _ in range(n_steps):
        step_fn()
    duration_s = (time.perf_counter_ns() - t0) / 1e9
    total_uj = _elapsed_uj(domains, start, _sum_uj(domains))

    idle_scaled_uj = idle_uj * (duration_s / idle_duration_s) if idle_duration_s > 0 else 0.0
    net_uj = max(total_uj - idle_scaled_uj, 0.0)

    total_j, net_j = total_uj / 1e6, net_uj / 1e6
    per_inference_mj = (total_uj / 1e3) / n_steps
    net_per_inference_mj = (net_uj / 1e3) / n_steps
    return {
        "status": "measured",
        "backend": "intel_rapl",
        "domains": [d.name for d in domains],
        "n_steps": n_steps,
        "duration_s": duration_s,
        "idle_duration_s": idle_duration_s,
        "total_energy_j": total_j,
        "idle_energy_j": idle_scaled_uj / 1e6,
        "net_energy_j": net_j,
        "energy_per_inference_mj": per_inference_mj,
        "net_energy_per_inference_mj": net_per_inference_mj,
        "samples_per_joule": (n_steps / total_j) if total_j > 0 else None,
        "net_samples_per_joule": (n_steps / net_j) if net_j > 0 else None,
        # EDP на один вывод: (net энергия одного шага, Дж) × (p50 задержка, с).
        "energy_delay_product_j_s": (net_per_inference_mj / 1e3) * (p50_ns / 1e9),
        "note": (
            "package-domain energy; idle baseline of equal duration subtracted "
            "for the net values. Not a per-model isolated measurement."
        ),
    }
