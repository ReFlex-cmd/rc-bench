"""Энергия измеряется, а не оценивается. Тесты работают на подставном sysfs:
настоящий RAPL нельзя воспроизвести в CI, а вот арифметику окна, обработку
переполнения счётчика и честный отказ при отсутствии доступа — можно и нужно."""
import time
from pathlib import Path

import pytest

from rc_bench.profiling import energy as energy_mod


def _fake_rapl(tmp_path: Path, *, energy_uj: int = 1_000_000, readable: bool = True) -> Path:
    root = tmp_path / "powercap"
    domain = root / "intel-rapl:0"
    domain.mkdir(parents=True)
    (domain / "name").write_text("package-0\n")
    (domain / "max_energy_range_uj").write_text("262143328850\n")
    counter = domain / "energy_uj"
    counter.write_text(f"{energy_uj}\n")
    if not readable:
        counter.chmod(0o000)
    return root


def test_discover_returns_named_domains(tmp_path):
    root = _fake_rapl(tmp_path)
    domains = energy_mod.discover_rapl_domains(root)
    assert [d.name for d in domains] == ["package-0"]
    assert domains[0].max_range_uj == 262143328850


def test_status_is_unavailable_without_powercap(tmp_path):
    status = energy_mod.energy_backend_status(tmp_path / "nothing-here")
    assert status["status"] == "unavailable"
    assert "powercap" in status["reason"]


def test_status_is_unavailable_when_counter_is_not_readable(tmp_path):
    root = _fake_rapl(tmp_path, readable=False)
    status = energy_mod.energy_backend_status(root)
    assert status["status"] == "unavailable"
    assert "readable" in status["reason"]


def test_status_is_available_when_counter_reads(tmp_path):
    status = energy_mod.energy_backend_status(_fake_rapl(tmp_path))
    assert status["status"] == "available"
    assert status["backend"] == "intel_rapl"
    assert status["domains"] == ["package-0"]


def test_counter_wraparound_is_not_reported_as_negative_energy(tmp_path):
    root = _fake_rapl(tmp_path)
    domain = energy_mod.discover_rapl_domains(root)[0]
    # Счётчик переполнился: конец меньше начала. max_energy_range_uj —
    # исключительный модуль (счётчик пробегает 0..range-1 и сбрасывается),
    # поэтому полный цикл равен max_range_uj без поправки на единицу.
    delta_uj = energy_mod.counter_delta_uj(
        start_uj=domain.max_range_uj - 1_000, end_uj=2_000, domain=domain
    )
    assert delta_uj == pytest.approx(3_000)


def test_measure_energy_reports_unavailable_instead_of_failing(tmp_path):
    result = energy_mod.measure_energy(
        lambda: None, p50_ns=1_000.0, min_duration_s=0.01, min_steps=10,
        root=tmp_path / "nothing-here",
    )
    assert result["status"] == "unavailable"
    assert "energy_per_inference_mj" not in result


def test_measure_energy_derives_window_from_measured_latency(tmp_path, monkeypatch):
    root = _fake_rapl(tmp_path)
    # Чтения: 1 разведочное (energy_backend_status внутри measure_energy),
    # затем idle-старт/финиш и измерение-старт/финиш. Задаём дельты явно
    # (500 мкДж на простое, 1500 мкДж на измерении) вместо "поровну на
    # каждое чтение" — иначе, когда окна почти равны по длительности (как в
    # этом тесте, без до-прогона), фиксированная-на-чтение дельта делает
    # idle-базовую линию неотличимой от полной энергии и net схлопывается в
    # 0 независимо от того, правильно ли работает вычитание.
    schedule = [0, 0, 500, 500, 2_000]
    reads = {"n": 0}
    real_read = energy_mod.read_domain_uj

    def fake_read(domain):
        value = schedule[reads["n"]]
        reads["n"] += 1
        return value

    monkeypatch.setattr(energy_mod, "read_domain_uj", fake_read)
    calls = {"n": 0}

    # step_fn стоит примерно столько же, сколько заявляет p50_ns (1мс), так
    # что начальная оценка окна оправдывается сама и до-прогон не нужен —
    # проверяем именно базовый случай, «оценка попала».
    def step_fn():
        calls["n"] += 1
        time.sleep(0.0011)

    result = energy_mod.measure_energy(
        step_fn, p50_ns=1_000_000.0, min_duration_s=0.05, min_steps=10, root=root
    )
    assert result["status"] == "measured"
    # min_duration_s / p50 = 0.05с / 1мс = 50 шагов, что больше min_steps=10.
    assert result["n_steps"] == 50
    assert calls["n"] == 50
    assert result["p50_ns"] == 1_000_000.0
    assert result["min_duration_s"] == 0.05
    assert result["window_target_met"] is True
    assert result["duration_s"] >= result["min_duration_s"]
    assert result["energy_per_inference_mj"] > 0
    assert result["samples_per_joule"] > 0
    assert result["energy_delay_product_j_s"] > 0
    assert real_read is not energy_mod.read_domain_uj  # sanity: патч применился


def _load_driven_counter(monkeypatch, *, per_step_uj: int = 10) -> dict:
    """Счётчик, растущий от нагрузки, а не от времени.

    Статический файл не годится: с ним счётчик не сдвигается за окно, и
    measure_energy теперь честно отвечает "unavailable" — ровно так же, как на
    машине, где окно короче периода обновления. Здесь моделируется то, что
    делает настоящий RAPL: простой не тратит ничего, шаги тратят.
    """
    state = {"steps": 0}
    monkeypatch.setattr(
        energy_mod, "read_domain_uj", lambda domain: 1_000_000 + state["steps"] * per_step_uj
    )
    return state


def test_measure_energy_extends_window_when_estimate_undershoots(tmp_path, monkeypatch):
    root = _fake_rapl(tmp_path)
    state = _load_driven_counter(monkeypatch)
    calls = {"n": 0}

    def step_fn():
        calls["n"] += 1
        state["steps"] += 1

    # p50_ns сильно завышен относительно реальной (почти нулевой) стоимости
    # step_fn, поэтому начальная оценка числа шагов не наберёт min_duration_s
    # — модуль обязан до-прогнать шаги по наблюдённой стоимости, пока не
    # наберёт окно.
    result = energy_mod.measure_energy(
        step_fn, p50_ns=1e5, min_duration_s=0.05, min_steps=1, root=root,
        max_steps=5_000_000,
    )
    assert result["status"] == "measured"
    assert result["window_target_met"] is True
    assert result["duration_s"] >= result["min_duration_s"]
    # Начальная оценка (0.05с / 100мкс = 500 шагов) заведомо недостаточна.
    assert result["n_steps"] > 500
    assert calls["n"] == result["n_steps"]


def test_measure_energy_reports_unmet_window_when_step_cap_reached(tmp_path, monkeypatch):
    root = _fake_rapl(tmp_path)
    state = _load_driven_counter(monkeypatch)

    def step_fn():
        state["steps"] += 1

    # max_steps специально мал: даже с до-прогоном окно не наберёт
    # min_duration_s. Это должно быть видно потребителю явно, а не
    # маскироваться под полноценное "measured".
    result = energy_mod.measure_energy(
        step_fn, p50_ns=1e5, min_duration_s=0.05, min_steps=1, root=root,
        max_steps=50,
    )
    assert result["status"] == "measured"
    assert result["window_target_met"] is False
    assert result["n_steps"] == 50
    assert result["duration_s"] < result["min_duration_s"]


def test_measure_energy_refuses_a_cap_below_its_own_minimum(tmp_path):
    """`window_target_met` говорит только о длительности окна. Потолок ниже
    min_steps молча урезал бы вторую половину инварианта, и запись вышла бы
    с status='measured' без единого признака усечения."""
    root = _fake_rapl(tmp_path)

    with pytest.raises(ValueError, match="below min_steps"):
        energy_mod.measure_energy(
            lambda: None,
            p50_ns=1e5,
            min_duration_s=0.05,
            min_steps=1000,
            root=root,
            max_steps=500,
        )


def test_a_counter_that_never_advanced_is_not_a_measurement(tmp_path):
    """Окно короче периода обновления счётчика даёт нуль. Нуль, записанный как
    измеренная энергия, — худший исход: он выглядит числом и утверждает, что
    модель ничего не потребляет."""
    root = _fake_rapl(tmp_path)  # статический файл: счётчик не сдвинется

    result = energy_mod.measure_energy(
        lambda: None, p50_ns=1e5, min_duration_s=0.01, min_steps=1, root=root
    )

    assert result["status"] == "unavailable"
    assert "did not advance" in result["reason"]
    assert "net_energy_per_inference_mj" not in result


def test_net_energy_below_the_idle_baseline_is_not_a_measurement(tmp_path, monkeypatch):
    """Потребление модели утонуло в разбросе базовой линии: числа на один
    вывод из такого окна не получить, и ноль вместо него был бы шумом,
    выданным за результат."""
    root = _fake_rapl(tmp_path)
    # Порядок чтений: проверка доступности, начало/конец простоя,
    # начало/конец окна под нагрузкой. Простой «тратит» 10 мДж, нагрузка —
    # 0.1 мДж, то есть на два порядка меньше базовой линии.
    sequence = iter([0, 0, 10_000, 10_000, 10_100])

    def read(domain):
        return next(sequence, 10_100)

    monkeypatch.setattr(energy_mod, "read_domain_uj", read)

    result = energy_mod.measure_energy(
        lambda: None, p50_ns=1e5, min_duration_s=0.01, min_steps=1, root=root
    )

    assert result["status"] == "unavailable"
    assert "not resolvable" in result["reason"]


def test_a_zero_length_window_is_refused(tmp_path):
    """`window_target_met` сравнивает длительность с целью: при нулевой цели
    любое измерение «достигает» её тривиально."""
    root = _fake_rapl(tmp_path)

    with pytest.raises(ValueError, match="not a window"):
        energy_mod.measure_energy(
            lambda: None, p50_ns=1e5, min_duration_s=0.0, min_steps=1, root=root
        )
