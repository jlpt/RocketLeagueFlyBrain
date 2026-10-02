"""A small random connectome with the same structure/groups as the real one (for tests and smoke runs)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .graph import BrainGraph

GROUP_SIZES = {
    "in_visual": 64, "in_self": 32, "in_internal": 16, "out_dn": 40, "out_mn": 20,
    "dan_reward": 8, "dan_punish": 4, "kc": 50, "mbon": 10, "giant_fiber": 2,
}


def synthetic_graph(n: int = 600, avg_degree: int = 20, n_types: int = 40, seed: int = 0) -> BrainGraph:
    rng = np.random.default_rng(seed)
    nnz = n * avg_degree
    pre = rng.integers(0, n, nnz)
    post = rng.integers(0, n, nnz)
    keep = pre != post
    syn = rng.geometric(0.2, nnz) + 4
    W = sp.csr_matrix((syn[keep].astype(np.int32), (post[keep], pre[keep])), shape=(n, n))
    W.sum_duplicates()
    W.sort_indices()
    perm = rng.permutation(n)
    groups, k = {}, 0
    for name, size in GROUP_SIZES.items():
        groups[name] = np.sort(perm[k:k + size])
        k += size
    superclass = np.array(["cb_intrinsic"] * n, dtype=object)
    superclass[groups["out_dn"]] = "descending_neuron"
    superclass[groups["out_mn"]] = "vnc_motor"
    superclass[groups["in_visual"]] = "visual_projection"
    cell_class = np.array([""] * n, dtype=object)
    cell_class[groups["kc"]] = "Kenyon_Cell"
    cell_class[groups["mbon"]] = "MBON"
    cell_class[np.concatenate([groups["dan_reward"], groups["dan_punish"]])] = "DAN"
    type_id = rng.integers(0, n_types, n).astype(np.int32)
    type_names = np.array([f"T{i}" for i in range(n_types)])
    nt = rng.choice(["acetylcholine", "gaba", "glutamate"], n, p=[0.65, 0.2, 0.15])
    sign = np.where(nt == "acetylcholine", 1.0, -1.0).astype(np.float32)
    return BrainGraph(
        body_id=np.arange(n, dtype=np.int64), superclass=superclass.astype(str), cell_class=cell_class.astype(str),
        cell_type=type_names[type_id], nt=nt, sign=sign, type_id=type_id, type_names=type_names,
        indptr=W.indptr.astype(np.int64), indices=W.indices.astype(np.int32), syn=W.data.astype(np.int32),
        groups=groups, meta={"dataset": "synthetic"},
    )
