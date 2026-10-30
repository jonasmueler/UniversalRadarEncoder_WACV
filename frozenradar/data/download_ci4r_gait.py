"""Download the 77 GHz raw captures of CI4R-Gait from its public GCS bucket."""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_ZIP_URL = (
    "https://storage.googleapis.com/cross-frequency_dataset/Raw%20Data/raw_data.zip"
)
_PREFIX = "Cross-frequency/77ghz/"


class _HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over an HTTP resource via Range requests.

    Args:
        url: URL of the resource; it must report ``Content-Length``.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._pos = 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self._size = int(head.headers["Content-Length"])

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move the read position.

        Args:
            offset: Byte offset.
            whence: 0 = from the start, 1 = from the current position, 2 = from the end.

        Returns:
            The new position.
        """
        self._pos = (
            offset if whence == 0
            else self._pos + offset if whence == 1
            else self._size + offset
        )
        return self._pos

    def tell(self) -> int:
        """Return the current read position."""
        return self._pos

    def seekable(self) -> bool:
        """Return True: the resource supports random access."""
        return True

    def readable(self) -> bool:
        """Return True: the resource is readable."""
        return True

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes with one Range request.

        Args:
            size: Number of bytes; negative reads to the end.

        Returns:
            The bytes read.
        """
        if size is None or size < 0:
            size = self._size - self._pos
        if size == 0:
            return b""
        end = min(self._pos + size, self._size) - 1
        req = urllib.request.Request(
            self._url, headers={"Range": f"bytes={self._pos}-{end}"}
        )
        data = urllib.request.urlopen(req).read()
        self._pos += len(data)
        return data


def download_multi3_77ghz(output_dir: Path) -> None:
    """Extract every non-empty 77 GHz .bin into ``output_dir``.

    Members are skipped when a local file of the exact expected size already
    exists, so the download is resumable.

    Args:
        output_dir: Destination root; activity subfolders are created under it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(_HTTPRangeFile(_ZIP_URL))
    members = [
        i for i in zf.infolist()
        if _PREFIX in i.filename and i.filename.endswith(".bin") and i.file_size > 0
    ]
    logger.info("77 GHz .bin members: %d", len(members))

    done = 0
    for info in members:
        rel = info.filename.split(_PREFIX, 1)[1]  # <activity>/<file>.bin
        dst = output_dir / rel
        if dst.exists() and dst.stat().st_size == info.file_size:
            done += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(dst, "wb") as out:
            while chunk := src.read(1 << 22):
                out.write(chunk)
        done += 1
        if done % 25 == 0:
            logger.info("  %d/%d extracted", done, len(members))
    logger.info("Done: %d/%d files in %s", done, len(members), output_dir)


def main() -> None:
    """Download the CI4R-Gait 77 GHz captures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    download_multi3_77ghz(args.output_dir)


if __name__ == "__main__":
    main()
