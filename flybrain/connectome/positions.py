"""3D neuron coordinates for geometry-based (HyperNEAT/CPPN) encodings.

Soma positions come from the male CNS annotations (8 nm voxels). Sensory
neurons have their cell bodies outside the CNS, so they are placed at the
synapse-weighted centroid of the neurons they connect to, i.e. where their
axons actually terminate. Coordinates are normalised so that:

  u: left (+) / right (-) of the midline, in [-1, 1]  (mirror symmetry = |u|)
  v: dorsal / ventral
  w: head (brain) -> tail (ventral nerve cord)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .download import raw_paths
from .graph import BrainGraph

MIDLINE_X = 48500.0
CENTER = np.array([MIDLINE_X, 37000.0, 72000.0])
SCALE = np.array([45000.0, 32000.0, 62000.0])


def soma_positions(graph: BrainGraph, raw_dir: str | Path = "data/raw") -> np.ndarray:
    ann = pd.read_feather(raw_paths(raw_dir)["annotations"], columns=["bodyId", "somaLocation", "tosomaLocation"])
    ann = ann.set_index("bodyId").reindex(graph.body_id)
    pos = np.full((graph.n, 3), np.nan)
    for col in ("somaLocation", "tosomaLocation"):
        vals = ann[col].values
        for i, v in enumerate(vals):
            if np.isnan(pos[i, 0]) and v is not None and not (isinstance(v, float) and np.isnan(v)):
                pos[i] = np.asarray(v, dtype=float)[:3]
    # place soma-less neurons at the weighted centroid of their partners (two passes)
    W = graph.csr().astype(np.float64)
    A = (W + W.T).tocsr()
    for _ in range(3):
        missing = np.isnan(pos[:, 0])
        if not missing.any():
            break
        known = np.where(~missing, 1.0, 0.0)
        P = np.nan_to_num(pos)
        num = A[missing] @ (P * known[:, None])
        den = A[missing] @ known
        ok = den > 0
        idx = np.where(missing)[0][ok]
        pos[idx] = num[ok] / den[ok, None]
    pos[np.isnan(pos[:, 0])] = CENTER
    return ((pos - CENTER) / SCALE).astype(np.float32)


def add_positions(graph_path: str | Path, raw_dir: str | Path = "data/raw", out: str | Path | None = None) -> BrainGraph:
    g = BrainGraph.load(graph_path)
    g.pos = soma_positions(g, raw_dir)
    g.save(out or graph_path)
    return g


def neuron_features(graph: BrainGraph) -> np.ndarray:
    """Per-neuron CPPN substrate coordinates: |u|, v, w, side, sign (exc/inh), layer.

    ``layer`` places each neuron on a sensory (-1) -> central (0) -> descending (0.5)
    -> motor (1) axis, the substrate's "depth" coordinate.
    """
    if getattr(graph, "pos", None) is None:
        raise ValueError("graph has no neuron positions; run `python -m flybrain positions` first")
    u, v, w = graph.pos[:, 0], graph.pos[:, 1], graph.pos[:, 2]
    side = np.sign(u)
    layer = np.zeros(graph.n, np.float32)
    sc = graph.superclass.astype(str)
    layer[np.char.find(sc, "sensory") >= 0] = -1.0
    layer[sc == "visual_projection"] = -0.5
    layer[sc == "ascending_neuron"] = -0.5
    layer[sc == "descending_neuron"] = 0.5
    layer[np.isin(sc, ["vnc_motor", "cb_motor"])] = 1.0
    return np.stack([np.abs(u), v, w, side, graph.sign, layer], axis=1).astype(np.float32)
