"""Build a BrainGraph from the male CNS flat-connectome tables.

Default choices (all overridable from the CLI):
  * neurons: proofread ``status == "Traced"`` bodies of the central brain and
    ventral nerve cord. The optic lobes (``ol_intrinsic`` / ``ol_sensory``,
    ~93k neurons) are excluded: we do not render images, so the photoreceptors
    are never driven. Visual information instead enters at the visual
    projection neurons (VPNs) -- the optic lobe's output layer -- and the
    optic lobe's processing time is accounted for as a fixed sensory delay
    (see ``flybrain.brain.model.TimingConfig``).
  * connections: >= 5 synapses (the conventional reliability threshold).
  * sign: Dale's law from the predicted presynaptic neurotransmitter.
    ACh -> excitatory; GABA, glutamate (GluCl), histamine -> inhibitory;
    monoamines (DA, 5-HT, OA) -> excitatory (they also gate plasticity).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf
import scipy.sparse as sp

from .download import raw_paths
from .graph import BrainGraph

NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 1.0,
    "serotonin": 1.0,
    "octopamine": 1.0,
    "tyramine": 1.0,
}

DEFAULT_EXCLUDE = ("ol_intrinsic", "ol_sensory")


def _resolve_nt(nt_df: pd.DataFrame) -> pd.Series:
    """consensus -> cell-type prediction -> per-body prediction -> ACh."""
    out = nt_df["consensus_nt"].astype("object")
    for col in ("celltype_predicted_nt", "predicted_nt"):
        bad = out.isna() | ~out.isin(list(NT_SIGN))
        out = out.where(~bad, nt_df[col].astype("object"))
    bad = out.isna() | ~out.isin(list(NT_SIGN))
    return out.where(~bad, "acetylcholine")


def _top_by_strength(candidates: np.ndarray, strength: np.ndarray, k: int) -> np.ndarray:
    if len(candidates) <= k:
        return np.sort(candidates)
    order = np.argsort(-strength[candidates], kind="stable")
    return np.sort(candidates[order[:k]])


def _startswith(arr: np.ndarray, prefix: str) -> np.ndarray:
    return np.char.startswith(arr.astype("U"), prefix)


def build_graph(
    raw_dir: str | Path = "data/raw",
    min_synapses: int = 5,
    exclude_superclasses: tuple[str, ...] = DEFAULT_EXCLUDE,
    n_visual: int = 2048,
    n_self: int = 1024,
    n_internal: int = 512,
    verbose: bool = True,
) -> BrainGraph:
    paths = raw_paths(raw_dir)
    log = print if verbose else (lambda *a, **k: None)

    ann = pd.read_feather(paths["annotations"])
    ann = ann[ann["status"] == "Traced"]
    ann = ann[~ann["superclass"].isin(exclude_superclasses)]
    log(f"[build] {len(ann)} traced neurons after excluding {exclude_superclasses}")

    nt = pd.read_feather(paths["neurotransmitters"])
    nt = nt.set_index("body")

    # ---- edges (streamed through arrow to keep memory reasonable)
    ids_arr = pa.array(ann["bodyId"].values)
    table = pf.read_table(paths["weights"], memory_map=True)
    mask = pc.and_(
        pc.and_(pc.is_in(table["body_pre"], value_set=ids_arr), pc.is_in(table["body_post"], value_set=ids_arr)),
        pc.greater_equal(table["weight"], min_synapses),
    )
    table = table.filter(mask)
    pre_b = table["body_pre"].to_numpy()
    post_b = table["body_post"].to_numpy()
    w = table["weight"].to_numpy().astype(np.int64)
    del table
    keep = pre_b != post_b  # drop autapses
    pre_b, post_b, w = pre_b[keep], post_b[keep], w[keep]
    log(f"[build] {len(w)} connections with >= {min_synapses} synapses ({w.sum()} synapses)")

    ann = ann.set_index("bodyId")
    superclass_all = ann["superclass"].fillna("unknown").astype(str)
    out_mask_body = superclass_all.isin(["descending_neuron", "vnc_motor", "cb_motor"])

    # ---- keep neurons that have edges and can reach an output neuron
    ids = np.unique(np.concatenate([pre_b, post_b]))
    pre = np.searchsorted(ids, pre_b)
    post = np.searchsorted(ids, post_b)
    n = len(ids)
    rev = sp.csr_matrix((np.ones(len(pre), np.int8), (post, pre)), shape=(n, n))  # post -> pre
    is_out = out_mask_body.reindex(ids).fillna(False).values.astype(bool)
    reach = is_out.copy()
    frontier = np.where(is_out)[0]
    while len(frontier):
        nxt = np.unique(rev[frontier].indices)
        nxt = nxt[~reach[nxt]]
        reach[nxt] = True
        frontier = nxt
    log(f"[build] pruning {n - reach.sum()} neurons that cannot influence motor output")
    sel = np.where(reach)[0]
    remap = -np.ones(n, np.int64)
    remap[sel] = np.arange(len(sel))
    ek = reach[pre] & reach[post]
    pre, post, w = remap[pre[ek]], remap[post[ek]], w[ek]
    ids = ids[sel]
    n = len(ids)

    W = sp.csr_matrix((w.astype(np.int32), (post, pre)), shape=(n, n))
    W.sum_duplicates()
    W.sort_indices()

    a = ann.reindex(ids)
    superclass = a["superclass"].fillna("unknown").astype(str).values
    cell_class = a["class"].fillna("").astype(str).values
    cell_type = a["type"].fillna("").astype(str).values

    nt_res = _resolve_nt(nt.reindex(ids))
    nt_names = nt_res.values.astype(str)
    sign = np.array([NT_SIGN.get(x, 1.0) for x in nt_names], dtype=np.float32)

    # indirect-encoding unit: the annotated cell type (fallback: superclass/class)
    unit = np.where(cell_type != "", cell_type, np.char.add(np.char.add(superclass.astype("U"), "/"), cell_class.astype("U")))
    type_names, type_id = np.unique(unit, return_inverse=True)

    out_strength = np.bincount(W.indices, weights=W.data, minlength=n)
    sc = superclass
    visual_c = np.where(sc == "visual_projection")[0]
    self_c = np.where(np.isin(cell_class, ["mechanosensory_proprioceptive", "mechanosensory"]))[0]
    internal_c = np.where(np.isin(cell_class, ["gustatory", "chemosensory", "hygrosensory", "thermosensory"]))[0]
    groups = {
        "in_visual": _top_by_strength(visual_c, out_strength, n_visual),
        "in_self": _top_by_strength(self_c, out_strength, n_self),
        "in_internal": _top_by_strength(internal_c, out_strength, n_internal),
        "out_dn": np.where(sc == "descending_neuron")[0],
        "out_mn": np.where(np.isin(sc, ["vnc_motor", "cb_motor"]))[0],
        "dan_reward": np.where((cell_class == "DAN") & _startswith(cell_type, "PAM"))[0],
        "dan_punish": np.where((cell_class == "DAN") & _startswith(cell_type, "PPL1"))[0],
        "kc": np.where(cell_class == "Kenyon_Cell")[0],
        "mbon": np.where(cell_class == "MBON")[0],
        "giant_fiber": np.where(cell_type == "DNp01")[0],
    }

    meta = {
        "dataset": "FlyEM male CNS v1.0 (Janelia/Google), CC-BY 4.0",
        "min_synapses": str(min_synapses),
        "excluded_superclasses": ",".join(exclude_superclasses),
    }
    g = BrainGraph(
        body_id=ids.astype(np.int64), superclass=superclass, cell_class=cell_class,
        cell_type=cell_type, nt=nt_names, sign=sign, type_id=type_id.astype(np.int32),
        type_names=type_names, indptr=W.indptr.astype(np.int64),
        indices=W.indices.astype(np.int32), syn=W.data.astype(np.int32),
        groups=groups, meta=meta,
    )
    try:
        from .positions import soma_positions
        g.pos = soma_positions(g, raw_dir)
    except Exception as e:  # positions are optional (only HyperNEAT needs them)
        log(f"[build] no soma positions: {e}")
    log(g.summary())
    return g


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the fly brain graph from male CNS tables")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/brain/male_cns.npz")
    p.add_argument("--min-synapses", type=int, default=5)
    p.add_argument("--include-optic-lobe", action="store_true")
    args = p.parse_args(argv)
    excl = () if args.include_optic_lobe else DEFAULT_EXCLUDE
    g = build_graph(args.raw_dir, args.min_synapses, excl)
    g.save(args.out)
    print(f"[build] saved {args.out}")


if __name__ == "__main__":
    main()
