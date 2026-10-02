"""Import and verify real arena meshes dumped with RLArenaCollisionDumper.

Usage (on a Windows PC with Rocket League):
  1. download RLArenaCollisionDumper from
     https://github.com/ZealanL/RLArenaCollisionDumper/releases
  2. start Rocket League, enter Free Play (a standard soccar arena)
  3. run the dumper; it writes ``.cmf`` files into ``./collision-meshes/``
  4. ``python -m flybrain import-meshes path/to/collision-meshes``
  5. ``export RS_COLLISION_MESHES=collision_meshes_real`` (or pass --meshes)

The 16 standard soccar meshes are recognised by the same hash RocketSim uses.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import numpy as np

# from RocketSim/src/RocketSim.cpp (MeshHashSet, SOCCAR)
SOCCAR_HASHES = {
    0xA160BAF9, 0x2811EEE8, 0xB81AC8B9, 0x760358D3, 0x73AE4940, 0x918F4A4E, 0x1F8EE550, 0x255BA8C1,
    0x14B84668, 0xEC759EBF, 0x94FB0D5C, 0xDEA07102, 0xBD4FBEA8, 0x39A47F63, 0x3D79D25D, 0xD84C7A68,
}
M32 = 0xFFFFFFFF


def read_cmf(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = Path(path).read_bytes()
    n_tris, n_verts = struct.unpack_from("<ii", data, 0)
    tris = np.frombuffer(data, "<i4", n_tris * 3, 8).reshape(n_tris, 3)
    verts = np.frombuffer(data, "<f4", n_verts * 3, 8 + 12 * n_tris).reshape(n_verts, 3)
    return tris, verts


def mesh_hash(tris: np.ndarray, verts: np.ndarray) -> int:
    """Python port of RocketSim's CollisionMeshFile::UpdateHash."""
    h = (len(verts) + len(tris) * len(verts)) & M32
    vals = verts[tris.reshape(-1)].reshape(-1).astype(np.float64)
    vals = (np.trunc(vals).astype(np.int64) & M32).tolist()
    for cur in vals:
        for _ in range(2):
            cur = (((cur >> 16) ^ cur) * 0x45D9F3B) & M32
        cur = ((cur >> 16) ^ cur) & M32
        h = (h ^ ((cur + 0x9E3779B9 + ((h << 6) & M32) + (h >> 2)) & M32)) & M32
    return h


def import_dumped_meshes(src: str | Path, out: str | Path = "collision_meshes_real") -> Path:
    src, out = Path(src), Path(out)
    files = sorted(src.rglob("*.cmf"))
    if not files:
        raise SystemExit(f"no .cmf files found under {src}")
    dest = out / "soccar"
    dest.mkdir(parents=True, exist_ok=True)
    found = set()
    for f in files:
        h = mesh_hash(*read_cmf(f))
        if h in SOCCAR_HASHES:
            found.add(h)
            shutil.copy(f, dest / f"mesh_{h:08x}.cmf")
        else:
            print(f"  skipping {f.name}: hash 0x{h:08x} is not a standard soccar mesh")
    print(f"imported {len(found)}/16 standard soccar meshes into {dest}")
    if len(found) < 16:
        print("  WARNING: incomplete arena; make sure you dumped from a standard soccar map in free play")
    else:
        print(f"  use them with: export RS_COLLISION_MESHES={out}")
    return out
