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
