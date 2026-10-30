"""Download the CubeLearn dataset from Hugging Face."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:  # optional extra
    snapshot_download = None

logger = logging.getLogger(__name__)


def download_cubelearn(output_dir: Path) -> None:
    """Download the CubeLearn dataset with ``huggingface_hub``.

    Args:
        output_dir: Root output directory; files are saved directly here.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    if snapshot_download is None:
        raise ImportError("huggingface_hub is not installed: pip install -e '.[data]'")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading CubeLearn dataset to %s", output_dir)

    snapshot_download(
        repo_id="zhaoymn/cubelearn",
        repo_type="dataset",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )

    npy_files = list(output_dir.rglob("*.npy"))
    logger.info("Downloaded %d .npy files to %s", len(npy_files), output_dir)


def main() -> None:
    """Download the CubeLearn dataset."""
    parser = argparse.ArgumentParser(
        description="Download CubeLearn dataset from HuggingFace.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for dataset files.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    download_cubelearn(Path(args.output_dir))


if __name__ == "__main__":
    main()
