"""Download the released checkpoints from Google Drive and verify their SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

try:
    import gdown
except ImportError:  # optional extra
    gdown = None

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights_manifest.json"


def sha256(path: Path) -> str:
    """Hash a file.

    Args:
        path: File path.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(entries: list[dict], dest: Path) -> int:
    """Fetch every missing or corrupt file and verify it.

    Args:
        entries: Manifest entries with ``file``, ``id``, ``bytes`` and ``sha256``.
        dest: Local weights directory.

    Returns:
        Number of files that failed verification.

    Raises:
        ImportError: If ``gdown`` is not installed.
    """
    if gdown is None:
        raise ImportError("gdown is not installed: pip install -e '.[weights]'")
    failed = 0
    for i, entry in enumerate(entries, 1):
        path = dest / entry["file"]
        if path.exists() and sha256(path) == entry["sha256"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[%d/%d] %s (%.1f MB)", i, len(entries), entry["file"], entry["bytes"] / 1e6)
        try:
            gdown.download(id=entry["id"], output=str(path), quiet=True)
        except gdown.exceptions.FileURLRetrievalError:
            logger.error("Google Drive refused %s; is the folder shared as 'Anyone with the "
                         "link'? Stopping.", entry["file"])
            return failed + len(entries) - i + 1
        if not path.exists() or sha256(path) != entry["sha256"]:
            logger.error("checksum mismatch or download failed: %s", entry["file"])
            failed += 1
    return failed


def main() -> None:
    """Download the requested models' checkpoints."""
    manifest = json.loads(MANIFEST.read_text())
    released = sorted({e["model"] for e in manifest["files"]})
    parser = argparse.ArgumentParser(description="Download the released checkpoints.")
    parser.add_argument(
        "models", nargs="+", metavar="MODEL",
        help=f"one of {', '.join(released)}, or 'all' for every file in the manifest",
    )
    parser.add_argument("--dest", type=Path, default=REPO / "weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    unreleased = sorted(set(args.models) - {*released, "all"})
    if unreleased:
        parser.error(
            f"no released weights for {', '.join(unreleased)}; "
            f"available: {', '.join(released)} (or 'all')"
        )

    wanted = set(released) if "all" in args.models else set(args.models)
    entries = [e for e in manifest["files"] if e["model"] in wanted]
    total = sum(e["bytes"] for e in entries)
    logger.info("%d files, %.1f GB -> %s (Drive folder: %s)", len(entries), total / 1e9,
                args.dest, manifest["drive_folder"])
    try:
        failed = download(entries, args.dest)
    except ImportError as exc:
        sys.exit(str(exc))
    if failed:
        sys.exit(f"{failed} files failed; rerun the same command to retry them")
    logger.info("all %d files verified", len(entries))


if __name__ == "__main__":
    main()
