"""Extract frozen-encoder embeddings from CI4R-Gait raw captures."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from frozenradar.data.multi3_dataset import Multi3VAEDataset
from frozenradar.extract_embeddings import extract_raw_tree
from frozenradar.utils.config import load_config, parse_overrides

logger = logging.getLogger(__name__)


def main() -> None:
    """Extract the embeddings of the CI4R-Gait raw tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, extra = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cfg = load_config(args.config, overrides=parse_overrides(extra))
    inner = Multi3VAEDataset(
        root=args.root,
        apply_fft=cfg.dataset.apply_fft,
        clutter_removal=cfg.dataset.clutter_removal,
    )
    extract_raw_tree(inner, cfg, args.checkpoint, args.root, args.output_dir)


if __name__ == "__main__":
    main()
