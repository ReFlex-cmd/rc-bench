"""Проверки обложки репозитория: то, что README обещает читателю, должно
существовать в дереве. Бейдж лицензии без файла лицензии — заявление без
артефакта, ровно то, что запрещает Definition of Done для документации
(docs/agent/BACKLOG.md)."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_license_file_exists_and_is_mit():
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.is_file(), "README ссылается на LICENSE, файла нет"
    text = license_path.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "2026" in text
    assert "Dmitry Komarov" in text


def test_readme_license_links_resolve():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\((LICENSE[^)]*)\)", readme)
    assert targets, "README больше не ссылается на LICENSE — тест потерял смысл"
    for target in targets:
        assert (REPO_ROOT / target).is_file(), f"битая ссылка на {target}"


def test_readme_separates_implemented_from_planned():
    """§2 описания проекта обещает читателю, что планируемое отделено от
    сделанного. Проверяется наличие самой разметки, а не формулировок:
    оценивать текст тест не может, а вот заметить исчезновение границы —
    может."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Статус реализации" in readme
    for marker in ("Реализовано", "Запланировано", "Недоступно"):
        assert marker in readme, f"в README нет раздела «{marker}»"


def test_readme_numbers_point_at_evidence_artifacts():
    """Любое число в README должно быть проверяемым: читатель обязан знать,
    из какого файла оно взято, иначе это утверждение без источника."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for artifact in (
        "reports/jmlc_2026/aggregates/matrix_table.csv",
        "reports/jmlc_2026/profiles/summary.json",
        "reports/jmlc_2026/selection.md",
    ):
        assert artifact in readme, f"{artifact} нигде не назван как источник чисел"


def test_env_example_matches_what_the_test_suite_expects():
    """`.env.example` — единственная инструкция, по которой поднимается стек.
    Если его учётки разойдутся с умолчаниями conftest, команда из README
    поднимет контейнер, к которому integration-тесты не подключатся."""
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")

    for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        value = next(
            line.split("=", 1)[1].strip()
            for line in example.splitlines()
            if line.startswith(f"{key}=")
        )
        assert f'"{key}": "{value}"' in conftest, (
            f"{key}={value} в .env.example не совпадает с умолчанием conftest"
        )


def test_readme_does_not_promise_a_build_system_that_is_absent():
    """README долгое время обещал `make demo` при отсутствующем Makefile —
    заявление без артефакта ровно того же класса, что бейдж лицензии без
    файла лицензии. Настоящая демонстрация одной командой существует, но
    зовётся иначе (`rcbench run configs/jmlc/demo.yaml`)."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    make_targets = re.findall(r"\bmake\s+([a-zA-Z][\w-]*)", readme)
    if not make_targets:
        return
    makefile = next(
        (
            REPO_ROOT / name
            for name in ("Makefile", "makefile", "GNUmakefile")
            if (REPO_ROOT / name).is_file()
        ),
        None,
    )
    assert makefile is not None, (
        f"README называет команды make {sorted(set(make_targets))}, "
        "но Makefile в репозитории нет"
    )
    defined = set(re.findall(r"^([a-zA-Z][\w-]*):", makefile.read_text(encoding="utf-8"), re.M))
    missing = sorted(set(make_targets) - defined)
    assert not missing, f"README называет несуществующие цели make: {missing}"


def test_documented_service_startup_waits_for_readiness():
    """`docker compose up -d` возвращает управление по факту старта контейнера,
    а не готовности PostgreSQL. На свежем томе initdb занимает ~12 с, и
    команда вида `up -d db redis && pytest -m integration` падает четырьмя
    ERROR с `ConnectionResetError` — читатель видит не «БД ещё не готова», а
    сломанный тест. `--wait` дожидается healthcheck из docker-compose.yml."""
    sources = {
        "README.md": (REPO_ROOT / "README.md").read_text(encoding="utf-8"),
        ".env.example": (REPO_ROOT / ".env.example").read_text(encoding="utf-8"),
    }
    found = False
    for name, text in sources.items():
        for line in text.splitlines():
            if "docker compose up -d" not in line or "db redis" not in line:
                continue
            found = True
            assert "--wait" in line, (
                f"{name}: «{line.strip()}» поднимает БД без --wait, "
                "integration-тесты стартуют раньше, чем PostgreSQL примет "
                "соединения"
            )
    assert found, "команда запуска db/redis нигде не документирована — тест потерял смысл"
