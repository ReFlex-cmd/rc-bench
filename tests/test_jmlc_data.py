from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from rc_bench.data.download import ChecksumMismatchError, download_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MANIFEST = (
    REPO_ROOT / "configs" / "jmlc" / "dataset_manifest.json"
)
RAW_FILENAME = "household_power_consumption.txt"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_local_manifest(tmp_path: Path, raw_contents: bytes) -> Path:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(RAW_FILENAME, raw_contents)

    archive_contents = archive_path.read_bytes()
    manifest = {
        "schema_version": 1,
        "dataset": {
            "id": "test-dataset",
            "name": "Local checksum fixture",
            "doi": "10.0000/example",
            "landing_page_url": "https://example.invalid/dataset",
        },
        "license": {
            "name": "Test license",
            "url": "https://example.invalid/license",
        },
        "source": {
            "url": archive_path.as_uri(),
            "archive_filename": archive_path.name,
            "archive_sha256": _sha256(archive_contents),
            "archive_size_bytes": len(archive_contents),
        },
        "raw_file": {
            "archive_member": RAW_FILENAME,
            "filename": RAW_FILENAME,
            "sha256": _sha256(raw_contents),
            "size_bytes": len(raw_contents),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_official_manifest_pins_source_license_and_checksums() -> None:
    manifest = json.loads(OFFICIAL_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["dataset"] == {
        "id": "uci-235",
        "name": "Individual Household Electric Power Consumption",
        "doi": "10.24432/C58K54",
        "landing_page_url": (
            "https://archive.ics.uci.edu/dataset/235/"
            "individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption"
        ),
    }
    assert manifest["license"] == {
        "name": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "url": "https://creativecommons.org/licenses/by/4.0/",
    }
    assert manifest["source"]["url"].startswith("https://archive.ics.uci.edu/")
    assert manifest["source"]["archive_sha256"] == (
        "9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff"
    )
    assert manifest["raw_file"]["sha256"] == (
        "4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7"
    )


def test_download_dataset_verifies_and_extracts_local_archive(tmp_path: Path) -> None:
    raw_contents = b"Date;Time;Global_active_power\n16/12/2006;17:24:00;4.216\n"
    manifest_path = _make_local_manifest(tmp_path, raw_contents)
    output_dir = tmp_path / "raw"

    raw_path = download_dataset(manifest_path, output_dir)

    assert raw_path == output_dir / RAW_FILENAME
    assert raw_path.read_bytes() == raw_contents
    assert (output_dir / "source.zip").is_file()

    # A verified raw file makes the command deterministic and network-free on rerun.
    (tmp_path / "source.zip").unlink()
    assert download_dataset(manifest_path, output_dir) == raw_path


def test_download_dataset_rejects_archive_checksum_mismatch(tmp_path: Path) -> None:
    manifest_path = _make_local_manifest(tmp_path, b"valid fixture")
    (tmp_path / "source.zip").write_bytes(b"corrupted archive")
    output_dir = tmp_path / "raw"

    with pytest.raises(
        ChecksumMismatchError,
        match=r"archive SHA-256 mismatch.*expected [0-9a-f]{64}.*actual [0-9a-f]{64}",
    ):
        download_dataset(manifest_path, output_dir)

    assert not (output_dir / "source.zip").exists()
    assert not (output_dir / RAW_FILENAME).exists()


def test_download_dataset_rejects_raw_file_checksum_mismatch(tmp_path: Path) -> None:
    manifest_path = _make_local_manifest(tmp_path, b"valid fixture")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw_file"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "raw"

    with pytest.raises(
        ChecksumMismatchError,
        match=r"raw file SHA-256 mismatch.*expected ffff.*actual [0-9a-f]{64}",
    ):
        download_dataset(manifest_path, output_dir)

    assert not (output_dir / RAW_FILENAME).exists()


def test_raw_dataset_path_is_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "data/raw/household_power_consumption.txt"],
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0
