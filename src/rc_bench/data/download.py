"""Download and verify a dataset described by a pinned JSON manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COPY_CHUNK_BYTES = 1024 * 1024


class DatasetDownloadError(RuntimeError):
    """Base class for deterministic dataset acquisition failures."""


class DatasetManifestError(DatasetDownloadError):
    """Raised when a dataset manifest is missing or invalid."""


class ChecksumMismatchError(DatasetDownloadError):
    """Raised when downloaded or extracted bytes do not match the manifest."""


class SizeMismatchError(DatasetDownloadError):
    """Raised when downloaded or extracted byte counts do not match the manifest."""


@dataclass(frozen=True)
class DatasetManifest:
    """Validated subset of the JSON manifest needed for acquisition."""

    source_url: str
    archive_filename: str
    archive_sha256: str
    archive_size_bytes: int
    archive_member: str
    raw_filename: str
    raw_sha256: str
    raw_size_bytes: int

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        manifest_path = Path(path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DatasetManifestError(f"manifest does not exist: {manifest_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetManifestError(
                f"cannot read JSON manifest {manifest_path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise DatasetManifestError("manifest root must be a JSON object")
        if payload.get("schema_version") != 1:
            raise DatasetManifestError("manifest schema_version must equal 1")

        dataset = _required_object(payload, "dataset")
        for field in ("id", "name", "doi", "landing_page_url"):
            _required_string(dataset, field, section="dataset")
        license_info = _required_object(payload, "license")
        for field in ("name", "url"):
            _required_string(license_info, field, section="license")

        source = _required_object(payload, "source")
        raw_file = _required_object(payload, "raw_file")

        source_url = _required_string(source, "url", section="source")
        scheme = urlparse(source_url).scheme.lower()
        if scheme not in {"https", "file"}:
            raise DatasetManifestError(
                "source.url must use https (or file for local/offline fixtures)"
            )

        archive_filename = _safe_filename(
            _required_string(source, "archive_filename", section="source"),
            field="source.archive_filename",
        )
        archive_member = _required_string(
            raw_file, "archive_member", section="raw_file"
        )
        member_path = PurePosixPath(archive_member)
        if (
            member_path == PurePosixPath(".")
            or member_path.is_absolute()
            or ".." in member_path.parts
        ):
            raise DatasetManifestError("raw_file.archive_member must be a safe ZIP path")

        raw_filename = _safe_filename(
            _required_string(raw_file, "filename", section="raw_file"),
            field="raw_file.filename",
        )

        return cls(
            source_url=source_url,
            archive_filename=archive_filename,
            archive_sha256=_required_sha256(
                source, "archive_sha256", section="source"
            ),
            archive_size_bytes=_required_size(
                source, "archive_size_bytes", section="source"
            ),
            archive_member=archive_member,
            raw_filename=raw_filename,
            raw_sha256=_required_sha256(raw_file, "sha256", section="raw_file"),
            raw_size_bytes=_required_size(
                raw_file, "size_bytes", section="raw_file"
            ),
        )


def _required_object(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise DatasetManifestError(f"manifest.{field} must be a JSON object")
    return value


def _required_string(
    payload: dict[str, Any],
    field: str,
    *,
    section: str,
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetManifestError(f"{section}.{field} must be a non-empty string")
    return value


def _required_sha256(
    payload: dict[str, Any],
    field: str,
    *,
    section: str,
) -> str:
    value = _required_string(payload, field, section=section)
    if not _SHA256_RE.fullmatch(value):
        raise DatasetManifestError(
            f"{section}.{field} must be a 64-character hexadecimal SHA-256"
        )
    return value.lower()


def _required_size(
    payload: dict[str, Any],
    field: str,
    *,
    section: str,
) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetManifestError(f"{section}.{field} must be a non-negative integer")
    return value


def _safe_filename(value: str, *, field: str) -> str:
    if Path(value).name != value or value in {".", ".."}:
        raise DatasetManifestError(f"{field} must be a filename without directories")
    return value


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    label: str,
) -> None:
    actual_sha256, actual_size = _hash_file(path)
    if actual_sha256 != expected_sha256:
        raise ChecksumMismatchError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"actual {actual_sha256}"
        )
    if actual_size != expected_size:
        raise SizeMismatchError(
            f"{label} size mismatch: expected {expected_size} bytes, "
            f"actual {actual_size} bytes"
        )


def _temporary_path(directory: Path, *, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".part", dir=directory)
    os.close(descriptor)
    return Path(name)


def _download_archive(
    manifest: DatasetManifest,
    output_dir: Path,
    *,
    timeout_seconds: float,
) -> Path:
    archive_path = output_dir / manifest.archive_filename
    if archive_path.exists():
        _verify_file(
            archive_path,
            expected_sha256=manifest.archive_sha256,
            expected_size=manifest.archive_size_bytes,
            label="archive",
        )
        return archive_path

    temporary_path = _temporary_path(output_dir, prefix=f".{archive_path.name}.")
    request = urllib.request.Request(
        manifest.source_url,
        headers={"User-Agent": "rc-bench-dataset-downloader/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with temporary_path.open("wb") as destination:
                while chunk := response.read(_COPY_CHUNK_BYTES):
                    destination.write(chunk)
        _verify_file(
            temporary_path,
            expected_sha256=manifest.archive_sha256,
            expected_size=manifest.archive_size_bytes,
            label="archive",
        )
        temporary_path.replace(archive_path)
    except ChecksumMismatchError:
        raise
    except SizeMismatchError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise DatasetDownloadError(
            f"failed to download dataset from {manifest.source_url}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive_path


def _extract_raw_file(
    archive_path: Path,
    manifest: DatasetManifest,
    output_dir: Path,
) -> Path:
    raw_path = output_dir / manifest.raw_filename
    temporary_path = _temporary_path(output_dir, prefix=f".{raw_path.name}.")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            try:
                member = archive.getinfo(manifest.archive_member)
            except KeyError as exc:
                raise DatasetDownloadError(
                    f"archive does not contain required member "
                    f"{manifest.archive_member!r}"
                ) from exc
            if member.is_dir():
                raise DatasetDownloadError(
                    f"archive member {manifest.archive_member!r} is a directory"
                )
            with archive.open(member) as source, temporary_path.open("wb") as destination:
                while chunk := source.read(_COPY_CHUNK_BYTES):
                    destination.write(chunk)
        _verify_file(
            temporary_path,
            expected_sha256=manifest.raw_sha256,
            expected_size=manifest.raw_size_bytes,
            label="raw file",
        )
        temporary_path.replace(raw_path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise DatasetDownloadError(f"failed to extract {archive_path}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return raw_path


def download_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    timeout_seconds: float = 120.0,
) -> Path:
    """Return a locally verified raw file, downloading and extracting if needed.

    Existing files are never trusted implicitly: their byte length and SHA-256
    are checked against the pinned manifest before they are reused.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    manifest = DatasetManifest.load(manifest_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    raw_path = destination / manifest.raw_filename
    if raw_path.exists():
        _verify_file(
            raw_path,
            expected_sha256=manifest.raw_sha256,
            expected_size=manifest.raw_size_bytes,
            label="raw file",
        )
        return raw_path

    archive_path = _download_archive(
        manifest,
        destination,
        timeout_seconds=timeout_seconds,
    )
    return _extract_raw_file(archive_path, manifest, destination)
