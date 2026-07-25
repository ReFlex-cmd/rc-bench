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
