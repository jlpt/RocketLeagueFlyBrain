"""Download the Janelia/Google male CNS connectome (v1.0) flat tables.

Data: FlyEM Male CNS connectome, https://male-cns.janelia.org/ (CC-BY 4.0).
Only three tables are needed (~1.1 GB total):
  * body annotations   (cell type, superclass, class, side, status)
  * body neurotransmitters (predicted NT per neuron -> synapse sign)
  * connectome weights (pre body, post body, synapse count)
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}


def raw_paths(raw_dir: str | Path) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    return {k: raw_dir / v for k, v in FILES.items()}


def download(raw_dir: str | Path = "data/raw", force: bool = False) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = raw_paths(raw_dir)
    for key, path in paths.items():
        if path.exists() and not force:
            print(f"[download] {path.name} already present ({path.stat().st_size / 1e6:.1f} MB)")
            continue
        url = f"{BASE_URL}/{path.name}"
        tmp = path.with_suffix(path.suffix + ".part")
        print(f"[download] {url}")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = resp.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
                    sys.stdout.flush()
        sys.stdout.write("\n")
        shutil.move(tmp, path)
    return paths


if __name__ == "__main__":
    download(os.environ.get("FLYBRAIN_RAW_DIR", "data/raw"))
