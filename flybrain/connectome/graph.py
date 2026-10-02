"""BrainGraph: a compact, model-ready view of a connectome.

Rows of the CSR matrix are *postsynaptic* neurons, columns are *presynaptic*
neurons, values are synapse counts. Synapse sign is a property of the
presynaptic neuron (Dale's law) and is stored per neuron in ``sign``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Named neuron groups the model relies on.
INPUT_GROUPS = ("in_visual", "in_self", "in_internal")
OUTPUT_GROUPS = ("out_dn", "out_mn")
MODULATORY_GROUPS = ("dan_reward", "dan_punish")


@dataclass
class BrainGraph:
    body_id: np.ndarray            # (N,) int64 FlyEM body ids
    superclass: np.ndarray         # (N,) str
    cell_class: np.ndarray         # (N,) str
    cell_type: np.ndarray          # (N,) str
    nt: np.ndarray                 # (N,) str predicted neurotransmitter
    sign: np.ndarray               # (N,) float32 in {-1,+1}
    type_id: np.ndarray            # (N,) int32 index into type_names (indirect encoding unit)
    type_names: np.ndarray         # (T,) str
    indptr: np.ndarray             # (N+1,) int64
    indices: np.ndarray            # (nnz,) int32 presynaptic index
    syn: np.ndarray                # (nnz,) int32 synapse count
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.body_id.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])

    @property
    def n_types(self) -> int:
        return int(self.type_names.shape[0])

    def csr(self, values: np.ndarray | None = None) -> sp.csr_matrix:
        v = self.syn.astype(np.float32) if values is None else values
        return sp.csr_matrix((v, self.indices, self.indptr), shape=(self.n, self.n))

    def signed_weights(self) -> np.ndarray:
        """Per-edge signed strength sign_pre * log1p(synapses) (unnormalised)."""
        return (self.sign[self.indices] * np.log1p(self.syn)).astype(np.float32)

    def post_index(self) -> np.ndarray:
        """Row (postsynaptic) index for every edge."""
        return np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.indptr))

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            body_id=self.body_id, superclass=self.superclass.astype("U"),
            cell_class=self.cell_class.astype("U"), cell_type=self.cell_type.astype("U"),
            nt=self.nt.astype("U"), sign=self.sign, type_id=self.type_id,
            type_names=self.type_names.astype("U"), indptr=self.indptr,
            indices=self.indices, syn=self.syn,
            meta_keys=np.array(list(self.meta.keys()), dtype="U"),
            meta_vals=np.array(list(self.meta.values()), dtype="U"),
        )
        for k, v in self.groups.items():
            arrays[f"group__{k}"] = v.astype(np.int32)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "BrainGraph":
        z = np.load(path, allow_pickle=False)
        groups = {k[len("group__"):]: z[k] for k in z.files if k.startswith("group__")}
        meta = dict(zip(z["meta_keys"].tolist(), z["meta_vals"].tolist())) if "meta_keys" in z.files else {}
        return cls(
            body_id=z["body_id"], superclass=z["superclass"], cell_class=z["cell_class"],
            cell_type=z["cell_type"], nt=z["nt"], sign=z["sign"].astype(np.float32),
            type_id=z["type_id"].astype(np.int32), type_names=z["type_names"],
            indptr=z["indptr"].astype(np.int64), indices=z["indices"].astype(np.int32),
            syn=z["syn"].astype(np.int32), groups=groups, meta=meta,
        )

    def summary(self) -> str:
        lines = [f"BrainGraph: {self.n} neurons, {self.nnz} connections, "
                 f"{int(self.syn.sum())} synapses, {self.n_types} cell types"]
        for k, v in self.meta.items():
            lines.append(f"  {k}: {v}")
        exc = (self.sign[self.indices] > 0).mean() if self.nnz else 0.0
        lines.append(f"  excitatory edge fraction: {exc:.3f}")
        for k, v in sorted(self.groups.items()):
            lines.append(f"  group {k:12s}: {len(v)} neurons")
        return "\n".join(lines)
