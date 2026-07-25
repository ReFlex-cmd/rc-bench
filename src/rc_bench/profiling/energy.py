"""Измерение энергии вывода через Intel RAPL (ENERGY-002).

Протокол (docs/PROJECT_CONTRACT.md, «Energy»):

- источник — powercap-интерфейс ядра, ``/sys/class/powercap/intel-rapl:*``;
  берутся домены верхнего уровня (package), а не их поддомены, чтобы не
  сложить одну и ту же энергию дважды;
- окно измерения не короче ``min_duration_s`` (по умолчанию 2 с) И не меньше
  ``min_steps`` шагов: разрешение счётчика ~61 мкДж и период обновления ~1 мс
  сопоставимы с длительностью одного шага (десятки микросекунд), поэтому
  измерять один вызов бессмысленно. Число шагов сперва *оценивается* по
  переданному ``p50_ns``, но эта оценка — из другого прохода (latency) и
  может ошибаться; поэтому фактическая длительность окна проверяется после
  прогона и, если её не хватило, окно расширяется по уже наблюдённой (а не
  исходной) стоимости шага, пока не наберётся ``min_duration_s`` или не
  упрёмся в ``max_steps``. Итог помечается ``window_target_met``, чтобы
  потребитель не принял урезанное окно за полноценное измерение только
  потому, что ``status == "measured"``;
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
from typing import Any, Callable, Dict, List, Tuple

RAPL_ROOT = Path("/sys/class/powercap")

# Верхняя граница на общее число шагов при расширении окна измерения.
# Ориентир — самый быстрый профиль проекта (persistence, ~0.27 мкс/шаг):
# 2 с / 0.27 мкс ≈ 7.4М шагов, поэтому 20М оставляет запас, но не даёт
# патологически быстрому step_fn крутиться неограниченно.
DEFAULT_MAX_STEPS = 20_000_000


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
    # Не-числовое содержимое energy_uj означало бы, что сломан драйвер/ядро,
    # а не то, что счётчик просто недоступен этому пользователю — это другой
    # класс отказа, чем OSError/PermissionError, которые обрабатывает
    # ``energy_backend_status``. "unavailable" в DEC-007 зарезервировано за
    # отказами доступа, поэтому здесь ValueError сознательно не глушится.
    return int(domain.energy_path.read_text().strip())


def counter_delta_uj(*, start_uj: int, end_uj: int, domain: RaplDomain) -> int:
    """Разность с учётом кольцевого переполнения счётчика.

    По документации powercap ``max_energy_range_uj`` — максимальное значение,
    которое счётчик может показать, то есть включительный максимум: на этой
    машине это 65532610987 мкДж = 0xFFFFFFFF × 15.258 мкДж (квант домена).
    Настоящий период переполнения на один квант больше, но квант на этом
    уровне неизвестен, поэтому ``max_range_uj`` берётся как его нижняя
    оценка: недоучёт составляет один квант на переполнение (~1.5e-11 от
    полного цикла) и не может превысить разрешение самого счётчика.
    """
    if end_uj >= start_uj:
        return end_uj - start_uj
    return (domain.max_range_uj - start_uj) + end_uj


def _discover_available(root: Path) -> Tuple[Dict[str, Any], List[RaplDomain]]:
    """Общий для ``energy_backend_status`` и ``measure_energy`` обход sysfs,
    чтобы измерение не делало тот же directory scan дважды на каждый вызов."""
    domains = discover_rapl_domains(root)
    if not domains:
        return (
            {
                "status": "unavailable",
                "backend": None,
                "reason": f"no powercap intel-rapl domains under {root}",
            },
            [],
        )
    try:
        for domain in domains:
            read_domain_uj(domain)
    except OSError as exc:
        return (
            {
                "status": "unavailable",
                "backend": None,
                "reason": (
                    f"intel-rapl counter {exc.filename} is not readable by this user "
                    "(root-only since CVE-2020-8694); see the plan's prerequisite"
                ),
            },
            [],
        )
    return (
        {
            "status": "available",
            "backend": "intel_rapl",
            "domains": [d.name for d in domains],
        },
        domains,
    )


def energy_backend_status(root: Path = RAPL_ROOT) -> Dict[str, Any]:
    status, _domains = _discover_available(root)
    return status


def _sum_uj(domains: List[RaplDomain]) -> List[int]:
    return [read_domain_uj(d) for d in domains]


def _elapsed_uj(domains: List[RaplDomain], start: List[int], end: List[int]) -> float:
    return float(
        sum(
            counter_delta_uj(start_uj=s, end_uj=e, domain=d)
            for d, s, e in zip(domains, start, end)
        )
    )


def _run_step_window(
    step_fn: Callable[[], Any],
    *,
    initial_steps: int,
    min_duration_s: float,
    max_steps: int,
) -> Tuple[int, float, bool]:
    """Прогоняет ``step_fn`` не менее ``initial_steps`` раз, затем сверяет
    фактическую длительность окна с ``min_duration_s`` и, если не хватило,
    до-прогоняет ещё — по стоимости шага, *уже наблюдённой* в этом самом
    прогоне (исходная оценка через ``p50_ns`` уже показала себя ненадёжной),
    пока не наберётся окно или не будет достигнут ``max_steps``.

    Возвращает ``(steps_run, duration_s, window_target_met)``.
    """
    if initial_steps < 1:
        raise ValueError(f"initial_steps must be >= 1, got {initial_steps}")
    if max_steps < initial_steps:
        # Иначе вторая половина инварианта («не меньше min_steps шагов») была
        # бы отброшена молча, а window_target_met следит только за временем и
        # такую усечённость не показал бы.
        raise ValueError(
            f"max_steps={max_steps} is below the requested {initial_steps} steps; "
            "the step-count half of the measurement window cannot be met"
        )
    steps_run = initial_steps
    t0 = time.perf_counter_ns()
    for _ in range(steps_run):
        step_fn()
    duration_s = (time.perf_counter_ns() - t0) / 1e9

    while duration_s < min_duration_s and steps_run < max_steps:
        observed_step_s = duration_s / steps_run
        if observed_step_s > 0:
            remaining_steps = int((min_duration_s - duration_s) / observed_step_s) + 1
        else:
            # Часы ещё не отличили прошедшее время от нуля (шаг быстрее
            # разрешения таймера) — удваиваем окно вместо того, чтобы делить
            # на ноль или сразу выбирать весь остаток бюджета: следующая
            # итерация уже получит измеримую оценку.
            remaining_steps = steps_run
        extra_steps = max(1, min(remaining_steps, max_steps - steps_run))
        for _ in range(extra_steps):
            step_fn()
        steps_run += extra_steps
        duration_s = (time.perf_counter_ns() - t0) / 1e9

    return steps_run, duration_s, duration_s >= min_duration_s


def measure_energy(
    step_fn: Callable[[], Any],
    *,
    p50_ns: float,
    min_duration_s: float = 2.0,
    min_steps: int = 1000,
    root: Path = RAPL_ROOT,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, Any]:
    """Энергия одного шага вывода ``step_fn``.

    ``p50_ns`` — уже измеренная медианная задержка шага: по ней выбирается
    начальное число шагов, чтобы окно измерения было длиннее периода
    обновления счётчика. Она же используется для energy-delay product.
    Фактическая длительность окна проверяется после прогона и при
    необходимости расширяется — см. :func:`_run_step_window`.

    Raises:
        ValueError: если ``max_steps`` не даёт набрать даже ``min_steps``.
            ``window_target_met`` следит только за длительностью окна, поэтому
            нарушенную половину инварианта («не меньше ``min_steps`` шагов»)
            иначе никто бы не заметил.
    """
    if min_duration_s <= 0:
        raise ValueError(
            f"min_duration_s={min_duration_s} is not a window: with a zero-length "
            "target every measurement trivially 'meets' it"
        )
    if max_steps < min_steps:
        raise ValueError(
            f"max_steps={max_steps} is below min_steps={min_steps}: the window "
            "cannot satisfy the step-count half of its own invariant"
        )

    status, domains = _discover_available(root)
    if status["status"] != "available":
        return status

    # Оценка по p50 может превысить потолок — это законно и отражается в
    # window_target_met; ниже min_steps опуститься нельзя (проверено выше).
    n_steps = max(int(min_steps), int((min_duration_s * 1e9) // max(p50_ns, 1.0)))
    n_steps = min(n_steps, max_steps)

    # Простой той же длительности — базовая линия статического потребления.
    idle_start = _sum_uj(domains)
    t0 = time.perf_counter_ns()
    time.sleep(min_duration_s)
    idle_duration_s = (time.perf_counter_ns() - t0) / 1e9
    idle_uj = _elapsed_uj(domains, idle_start, _sum_uj(domains))

    start = _sum_uj(domains)
    n_steps, duration_s, window_target_met = _run_step_window(
        step_fn,
        initial_steps=n_steps,
        min_duration_s=min_duration_s,
        max_steps=max_steps,
    )
    total_uj = _elapsed_uj(domains, start, _sum_uj(domains))

    idle_scaled_uj = idle_uj * (duration_s / idle_duration_s) if idle_duration_s > 0 else 0.0
    net_uj = max(total_uj - idle_scaled_uj, 0.0)

    total_j, net_j = total_uj / 1e6, net_uj / 1e6

    # Счётчик обновляется с периодом порядка миллисекунды. Если за всё окно он
    # не сдвинулся, измерения не было — а нуль, записанный как измеренная
    # энергия, это худший исход из возможных: он и выглядит числом, и
    # утверждает, что модель ничего не потребляет.
    if total_uj <= 0:
        return {
            "status": "unavailable",
            "backend": None,
            "reason": (
                f"intel-rapl counters did not advance over {duration_s:.4f} s "
                "(window shorter than the counter update period)"
            ),
        }
    # Net ≤ 0 означает, что потребление модели утонуло в разбросе базовой
    # линии: числа на один вывод из такого окна не получить, и подставлять
    # ноль вместо него значило бы выдать шум за результат.
    if net_uj <= 0:
        return {
            "status": "unavailable",
            "backend": None,
            "reason": (
                f"net energy is not resolvable: the load window drew "
                f"{total_j:.4f} J against an idle baseline of "
                f"{idle_scaled_uj / 1e6:.4f} J over the same duration"
            ),
        }

    per_inference_mj = (total_uj / 1e3) / n_steps
    net_per_inference_mj = (net_uj / 1e3) / n_steps
    return {
        "status": "measured",
        "backend": "intel_rapl",
        "domains": [d.name for d in domains],
        "p50_ns": p50_ns,
        "min_duration_s": min_duration_s,
        "window_target_met": window_target_met,
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
            "package-domain energy; an idle baseline of min_duration_s, scaled "
            "to the measurement window, is subtracted for the net values. "
            "Not a per-model isolated measurement."
        ),
    }
