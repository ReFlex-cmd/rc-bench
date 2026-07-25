"""Энергия измеряется, а не оценивается. Тесты работают на подставном sysfs:
настоящий RAPL нельзя воспроизвести в CI, а вот арифметику окна, обработку
переполнения счётчика и честный отказ при отсутствии доступа — можно и нужно."""
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
    # Счётчик переполнился: конец меньше начала.
    delta_uj = energy_mod.counter_delta_uj(
        start_uj=domain.max_range_uj - 1_000, end_uj=2_000, domain=domain
    )
    assert delta_uj == pytest.approx(3_001)


def test_measure_energy_reports_unavailable_instead_of_failing(tmp_path):
    result = energy_mod.measure_energy(
        lambda: None, p50_ns=1_000.0, min_duration_s=0.01, min_steps=10,
        root=tmp_path / "nothing-here",
    )
    assert result["status"] == "unavailable"
    assert "energy_per_inference_mj" not in result


def test_measure_energy_derives_window_from_measured_latency(tmp_path, monkeypatch):
    root = _fake_rapl(tmp_path)
    # Счётчик, растущий на 1 мДж за каждое чтение, даёт предсказуемую арифметику.
    reads = {"n": 0}
    real_read = energy_mod.read_domain_uj

    def fake_read(domain):
        reads["n"] += 1
        return 1_000 * reads["n"]

    monkeypatch.setattr(energy_mod, "read_domain_uj", fake_read)
    calls = {"n": 0}

    def step_fn():
        calls["n"] += 1

    result = energy_mod.measure_energy(
        step_fn, p50_ns=1_000_000.0, min_duration_s=0.05, min_steps=10, root=root
    )
    assert result["status"] == "measured"
    # min_duration_s / p50 = 0.05с / 1мс = 50 шагов, что больше min_steps=10.
    assert result["n_steps"] == 50
    assert calls["n"] >= 50
    assert result["energy_per_inference_mj"] > 0
    assert result["samples_per_joule"] > 0
    assert result["energy_delay_product_j_s"] > 0
    assert real_read is not energy_mod.read_domain_uj  # sanity: патч применился
