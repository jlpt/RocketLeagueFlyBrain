"""Procedurally generated approximation of the Rocket League soccar arena.

RocketSim needs the arena collision meshes, which normally have to be dumped
from a local Rocket League install (https://github.com/ZealanL/RLArenaCollisionDumper).
If you have them, point ``RS_COLLISION_MESHES`` / ``--meshes`` at that folder
and the real arena is used.

Otherwise we generate a close approximation so training works anywhere:
RocketSim itself adds the floor, ceiling and the two side walls as planes;
this module supplies the rest: back walls with goal mouths, 45-degree corner
walls (|x| + |y| = 8064), the goal boxes, and rounded floor/wall and
wall/ceiling transitions (radius 256 uu) so cars can drive up the walls.

Mesh file format (``.cmf``): int32 n_tris, int32 n_verts, int32[n_tris][3]
triangle indices, float32[n_verts][3] vertices in Bullet units (uu / 50).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

UU_TO_BT = 1.0 / 50.0

EXTENT_X = 4096.0
EXTENT_Y = 5120.0
HEIGHT = 2048.0
CORNER_SUM = 8064.0           # corner walls lie on |x| + |y| = 8064
GOAL_HALF_W = 893.0           # goal mouth spans |x| < 893
GOAL_H = 642.775
GOAL_DEPTH = 880.0
FILLET_R = 256.0


def _perimeter() -> np.ndarray:
    """Counter-clockwise (seen from above) arena outline with goal-post vertices."""
    cx = CORNER_SUM - EXTENT_Y   # x where corner meets back wall (2944)
    cy = CORNER_SUM - EXTENT_X   # y where corner meets side wall (3968)
    X, Y, G = EXTENT_X, EXTENT_Y, GOAL_HALF_W
    pts = [
        (X, -cy), (X, cy),            # +x side wall
        (cx, Y), (G, Y), (-G, Y), (-cx, Y),   # +y back wall (goal between +-G)
        (-X, cy), (-X, -cy),          # -x side wall
        (-cx, -Y), (-G, -Y), (G, -Y), (cx, -Y),  # -y back wall
    ]
    return np.array(pts, dtype=np.float64)


def _inward_normals(poly: np.ndarray) -> np.ndarray:
    nxt = np.roll(poly, -1, axis=0)
    e = nxt - poly
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return np.stack([-e[:, 1], e[:, 0]], axis=1)  # left normal of a CCW polygon = inward


def _offset(poly: np.ndarray, d: float) -> np.ndarray:
    """Miter-offset a convex CCW polygon inward by ``d``."""
    n_edge = _inward_normals(poly)              # normal of edge i (poly[i] -> poly[i+1])
    n_prev = np.roll(n_edge, 1, axis=0)         # normal of edge i-1 (ending at poly[i])
    s = n_prev + n_edge
    denom = 1.0 + np.sum(n_prev * n_edge, axis=1, keepdims=True)
    return poly + d * s / denom


def _profile(n_arc: int = 8) -> list[tuple[float, float]]:
    """(inset, z) profile from the floor, up the wall, onto the ceiling."""
    R, H = FILLET_R, HEIGHT
    prof = []
    for th in np.linspace(0, np.pi / 2, n_arc + 1):
        prof.append((R - R * np.sin(th), R - R * np.cos(th)))
    for z in (GOAL_H, H / 2):
        prof.append((0.0, z))
    for ph in np.linspace(0, np.pi / 2, n_arc + 1):
        prof.append((R - R * np.cos(ph), H - R + R * np.sin(ph)))
    prof.sort(key=lambda p: p[1])
    return prof


def _goal_edge_indices(poly: np.ndarray) -> set[int]:
    out = set()
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if abs(abs(a[1]) - EXTENT_Y) < 1e-6 and abs(a[1] - b[1]) < 1e-6 and max(abs(a[0]), abs(b[0])) <= GOAL_HALF_W + 1e-6:
            out.add(i)
    return out


def _quad(tris: list, a: int, b: int, c: int, d: int) -> None:
    tris.append((a, b, c))
    tris.append((a, c, d))


def build_walls() -> tuple[np.ndarray, np.ndarray]:
    poly = _perimeter()
    prof = _profile()
    goal_edges = _goal_edge_indices(poly)
    m = len(poly)
    verts, tris = [], []
    rings = []
    for inset, z in prof:
        ring = _offset(poly, inset)
        base = len(verts)
        verts.extend([(p[0], p[1], z) for p in ring])
        rings.append((base, z))
    for (b0, z0), (b1, z1) in zip(rings[:-1], rings[1:]):
        for i in range(m):
            j = (i + 1) % m
            if i in goal_edges and z1 <= GOAL_H + 1e-6:
                continue  # leave the goal mouth open (no ramp, no wall)
            _quad(tris, b0 + i, b0 + j, b1 + j, b1 + i)
    return np.array(verts, np.float64), np.array(tris, np.int32)


def build_goal(sign: float) -> tuple[np.ndarray, np.ndarray]:
    """Goal box behind the back wall at y = sign * 5120 (floor provided by the floor plane)."""
    G, H, Y0, Y1 = GOAL_HALF_W, GOAL_H, EXTENT_Y, EXTENT_Y + GOAL_DEPTH
    s = sign
    v = np.array([
        (-G, s * Y0, 0), (G, s * Y0, 0), (G, s * Y1, 0), (-G, s * Y1, 0),
        (-G, s * Y0, H), (G, s * Y0, H), (G, s * Y1, H), (-G, s * Y1, H),
    ], np.float64)
    t = []
    _quad(t, 3, 2, 6, 7)      # back of the net
    _quad(t, 0, 3, 7, 4)      # -x side
    _quad(t, 1, 2, 6, 5)      # +x side
    _quad(t, 4, 5, 6, 7)      # roof
    # a short strip of floor inside the goal in case the floor plane is ever offset
    _quad(t, 0, 1, 2, 3)
    return v, np.array(t, np.int32)


def write_cmf(path: Path, verts_uu: np.ndarray, tris: np.ndarray) -> None:
    v = (verts_uu * UU_TO_BT).astype("<f4")
    t = tris.astype("<i4")
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", len(t), len(v)))
        f.write(t.tobytes())
        f.write(v.tobytes())


def generate(out_dir: str | Path = "collision_meshes") -> Path:
    """Write an approximate soccar arena into ``out_dir/soccar`` and return ``out_dir``."""
    out_dir = Path(out_dir)
    soccar = out_dir / "soccar"
    soccar.mkdir(parents=True, exist_ok=True)
    v, t = build_walls()
    write_cmf(soccar / "flybrain_walls.cmf", v, t)
    for name, s in (("flybrain_goal_orange.cmf", 1.0), ("flybrain_goal_blue.cmf", -1.0)):
        gv, gt = build_goal(s)
        write_cmf(soccar / name, gv, gt)
    (out_dir / "GENERATED_BY_FLYBRAIN").write_text(
        "Approximate soccar arena generated by flybrain.sim.arena_mesh. "
        "Replace with meshes dumped by RLArenaCollisionDumper for the exact arena.\n")
    return out_dir


if __name__ == "__main__":
    print(generate())
