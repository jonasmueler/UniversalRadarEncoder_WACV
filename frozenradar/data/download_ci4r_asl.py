"""Download the CI4R-ASL release from its public GCS bucket."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE = "https://storage.googleapis.com/asl_sequential_thms2021_dataset"
_API = (
    "https://storage.googleapis.com/storage/v1/b/asl_sequential_thms2021_dataset/o"
    "?fields=items(name,size)&maxResults=1000"
)
_LABELS = "Processed Data.zip"


def main() -> None:
    """Mirror all raw zips and the label archive into the output directory.

    Raises:
        SystemExit: Always; the exit code is 1 if any download failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = json.load(urllib.request.urlopen(_API)).get("items", [])
    zips = [
        it for it in items
        if it["name"].endswith(".zip")
        and (it["name"].startswith("Raw Data/") or it["name"] == _LABELS)
    ]
    total = sum(int(it["size"]) for it in zips)
    logger.info("%d zip objects, %.2f TB total", len(zips), total / 1e12)

    failed: list[str] = []
    for it in zips:
        name, size = it["name"], int(it["size"])
        if name == _LABELS:
            local = args.out_dir / "processed_data.zip"
        else:
            group = name.split("/")[-2].replace(" ", "_").lower()
            local = args.out_dir / (group + "__" + name.split("/")[-1].replace(" ", "_"))
        if local.exists() and local.stat().st_size == size:
            logger.info("[skip] %s", local.name)
            continue
        url = _BASE + "/" + urllib.parse.quote(name)
        logger.info("[%s] %s (%.1fG) ...", time.strftime("%H:%M:%S"), local.name, size / 1e9)
        rc = subprocess.run(["curl", "-sfL", "-o", str(local), url]).returncode
        ok = rc == 0 and local.exists() and local.stat().st_size == size
        logger.info(
            "[%s] %s %s", time.strftime("%H:%M:%S"), local.name, "OK" if ok else f"FAILED rc={rc}"
        )
        if not ok:
            failed.append(local.name)

    logger.info("mirror done (%d failed: %s)", len(failed), failed)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
