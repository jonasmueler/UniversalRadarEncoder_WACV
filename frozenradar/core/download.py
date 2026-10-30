"""Resumable download of the mmRadPose dataset from Zenodo."""

from __future__ import annotations

import logging
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "mmradpose": {
        "zenodo_record": "14738837",
        "variants": {
            "raw_adc": {
                "filename": "mmRadPose_rawdata.tar",
                "size_hint": "~65.6 GB",
            },
        },
        "url": "https://zenodo.org/records/14738837",
        "license": "CC BY-SA 4.0",
    },
}

logger = logging.getLogger(__name__)


def download(
    dataset: str,
    data_dir: str | Path = "~/data",
    variant: str = "raw_adc",
    force: bool = False,
) -> Path:
    """Download a dataset from Zenodo.

    Supports resuming interrupted downloads automatically.

    Args:
        dataset: Dataset name (e.g. ``"mmradpose"``).
        data_dir: Local directory to store the data.
        variant: Which variant to download (``"raw_adc"``).
        force: Re-download even if files already exist.

    Returns:
        Path to the extracted dataset directory.

    Raises:
        KeyError: If ``dataset`` or ``variant`` is not in the registry.
        RuntimeError: If the download does not complete after all retries.
    """
    dataset = dataset.lower()
    if dataset not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{dataset}'. Available: {sorted(DATASET_REGISTRY)}")

    entry = DATASET_REGISTRY[dataset]
    if variant not in entry["variants"]:
        raise KeyError(
            f"Unknown variant '{variant}' for {dataset}. Available: {sorted(entry['variants'])}"
        )

    var_info = entry["variants"][variant]
    filename = var_info["filename"]
    record_id = entry["zenodo_record"]
    url = f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"

    dest_dir = Path(data_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / filename

    extract_name = filename
    for ext in (".tar.gz", ".tar.bz2", ".tar", ".zip"):
        if extract_name.endswith(ext):
            extract_name = extract_name[: -len(ext)]
            break
    extract_dir = dest_dir / extract_name

    # A leftover archive next to the extracted dir signals a failed extraction.
    if extract_dir.exists() and not archive_path.exists() and not force:
        logger.info("Dataset already exists at %s (use force=True to re-download)", extract_dir)
        return extract_dir

    if force:
        if archive_path.exists():
            archive_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

    logger.info(
        "Downloading %s (%s) from Zenodo record %s", filename, var_info["size_hint"], record_id
    )
    _download_with_resume(url, archive_path)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    logger.info("Extracting to %s", dest_dir)
    _extract(archive_path, dest_dir)

    archive_path.unlink()
    logger.info("Download complete: %s", extract_dir)
    return extract_dir


def _extract(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .zip or .tar archive.

    Args:
        archive_path: Path to the archive file.
        dest_dir: Directory to extract into.

    Raises:
        ValueError: If the archive format is not supported.
    """
    name = archive_path.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif name.endswith((".tar", ".tar.gz", ".tar.bz2", ".tgz")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {name}")


def _download_with_resume(url: str, dest: Path, max_retries: int = 10) -> None:
    """Download a file with automatic resume on connection failure.

    If ``dest`` already exists (partial download), resumes from where it left
    off using an HTTP Range request.  On connection drops the download is
    retried up to ``max_retries`` times with exponential back-off.

    Args:
        url: Source URL.
        dest: Destination file path.
        max_retries: Maximum number of attempts.

    Raises:
        RuntimeError: If the file is still incomplete after all attempts.
    """
    total: int | None = None

    for attempt in range(1, max_retries + 1):
        existing_size = dest.stat().st_size if dest.exists() else 0

        req = urllib.request.Request(url)
        if existing_size > 0:
            req.add_header("Range", f"bytes={existing_size}-")
            logger.info("Resuming download from %d bytes", existing_size)

        try:
            with urllib.request.urlopen(req) as response:
                remaining = int(response.headers.get("Content-Length", 0))

                if response.status == 206:
                    total = existing_size + remaining
                else:
                    total = remaining
                    existing_size = 0

                mode = "ab" if existing_size > 0 else "wb"
                with (
                    open(dest, mode) as f,
                    tqdm(
                        total=total,
                        initial=existing_size,
                        unit="B",
                        unit_scale=True,
                        desc=dest.name,
                    ) as pbar,
                ):
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            downloaded = dest.stat().st_size if dest.exists() else 0
            if total and downloaded >= total:
                break  # complete despite the error
            wait = min(2**attempt, 60)
            logger.warning(
                "Connection lost after %s bytes (attempt %d/%d): %s. Retrying in %ds...",
                downloaded,
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
            continue

        final_size = dest.stat().st_size
        if total and final_size < total:
            wait = min(2**attempt, 60)
            logger.warning(
                "Download incomplete: %d / %d bytes (attempt %d/%d). Retrying in %ds...",
                final_size,
                total,
                attempt,
                max_retries,
                wait,
            )
            time.sleep(wait)
            continue

        return

    final_size = dest.stat().st_size if dest.exists() else 0
    expected = f" / {total}" if total else ""
    raise RuntimeError(
        f"Download failed after {max_retries} attempts: got {final_size}{expected} bytes."
    )
