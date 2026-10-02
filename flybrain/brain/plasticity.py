"""Dopamine-gated synaptic plasticity (a three-factor learning rule).

Where: two populations of synapses are plastic, chosen for biological reasons:

  * Kenyon cell -> mushroom body output neuron (KC->MBON) synapses: the
    fly's canonical site of dopamine-dependent associative learning.
  * all synapses onto descending neurons (DNs): the bottleneck between brain
    and body, so changing them changes behaviour directly.

How: every plastic synapse keeps an eligibility trace of pre/post coincidence
(covariance form, so only *deviations* of postsynaptic activity count):

    e_ij  <- lambda * e_ij + r_j * (r_i - <r_i>)
    D      = DA(t) - <DA>                 (dopamine prediction error)
    dW_ij <- (1 - decay) * dW_ij + eta * D * e_ij

where DA(t) is the *activity of the fly's own dopaminergic neurons*:
mean rate of the reward PAM cluster minus the punishment PPL1 cluster. Game
reward is delivered as current into those DANs (positive reward -> PAM,
negative -> PPL1), so dopamine is a real network signal that can also be
predicted by the brain itself. With exploration noise this rule is a form of
node-perturbation policy gradient (REINFORCE), i.e. it genuinely improves
reward on its own.

Dale's law is respected: dW_ij is clipped to [-|w_ij|, +|w_ij|], so a
synapse can be silenced or doubled but never change sign.

eta, lambda and the reward->dopamine gain are "genes": the evolution stage
tunes them (learning to learn).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..connectome.graph import BrainGraph


@dataclass
class PlasticityConfig:
    eta: float = 0.02                 # learning rate
    trace_lambda: float = 0.9         # eligibility decay per step (~330 ms)
    decay: float = 1e-4               # slow return of dW to the genetic baseline
    da_gain: float = 2.0              # reward -> DAN current
    post_avg_tau: float = 30.0        # steps for the running postsynaptic average
    da_avg_tau: float = 60.0          # steps for the dopamine baseline
    sites: tuple = ("kc_mbon", "dn")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sites"] = list(self.sites)
        return d


def plastic_edges(graph: BrainGraph, sites=("kc_mbon", "dn")) -> np.ndarray:
    post = graph.post_index()
    pre = graph.indices
    sel = np.zeros(graph.nnz, bool)
    if "dn" in sites:
        m = np.zeros(graph.n, bool)
        m[graph.groups["out_dn"]] = True
        sel |= m[post]
    if "kc_mbon" in sites:
        kc = np.zeros(graph.n, bool)
        kc[graph.groups["kc"]] = True
        mb = np.zeros(graph.n, bool)
        mb[graph.groups["mbon"]] = True
        sel |= kc[pre] & mb[post]
    return np.where(sel)[0]


class DopaminePlasticity:
    """Batched plasticity state (one column per simulated brain)."""

    def __init__(self, brain, graph: BrainGraph, cfg: PlasticityConfig | None = None):
        self.cfg = cfg or PlasticityConfig()
        e = plastic_edges(graph, self.cfg.sites)
        self.edge = torch.from_numpy(e.astype(np.int64))
        self.pre = brain.col[self.edge]
        post = graph.post_index()[e].astype(np.int64)
        self.post = torch.from_numpy(post)
        self.w0 = brain.w_values[self.edge].abs()
        self.post_neurons, self.post_local = torch.unique(self.post, return_inverse=True)
        self.n = brain.n
        self.brain = brain
        self.batch = 0
        # per-column genes (can be overridden per ES member)
        self.eta = None
        self.lam = None
        self.da_gain = None

    @property
    def n_synapses(self) -> int:
        return int(self.edge.numel())

    def reset(self, batch: int) -> None:
        E = self.edge.numel()
        self.batch = batch
        self.elig = torch.zeros(E, batch)
        self.dW = torch.zeros(E, batch)
        self.post_avg = torch.zeros(len(self.post_neurons), batch)
        self.da_avg = torch.zeros(batch)
        c = self.cfg
        self.eta = torch.full((batch,), c.eta)
        self.lam = torch.full((batch,), c.trace_lambda)
        self.da_gain = torch.full((batch,), c.da_gain)

    def reset_columns(self, cols) -> None:
        self.elig[:, cols] = 0
        self.dW[:, cols] = 0
        self.post_avg[:, cols] = 0
        self.da_avg[cols] = 0

    def set_genes(self, eta=None, lam=None, da_gain=None) -> None:
        if eta is not None:
            self.eta = torch.as_tensor(eta, dtype=torch.float32).expand(self.batch).clone()
        if lam is not None:
            self.lam = torch.as_tensor(lam, dtype=torch.float32).expand(self.batch).clone()
        if da_gain is not None:
            self.da_gain = torch.as_tensor(da_gain, dtype=torch.float32).expand(self.batch).clone()

    # --------------------------------------------------------------- brain hooks
    def current(self, r: torch.Tensor, g_pre: torch.Tensor, g_post: torch.Tensor) -> torch.Tensor:
        """Extra synaptic current from the learned weight changes (N, B)."""
        contrib = self.dW * g_pre.index_select(0, self.pre) * r.index_select(0, self.pre)
        out = torch.zeros_like(r).index_add_(0, self.post, contrib)
        return out * g_post

    def reward_current(self, reward: torch.Tensor) -> torch.Tensor:
        """Inject game reward into the dopaminergic neurons (N, B)."""
        B = reward.shape[0]
        I = torch.zeros(self.n, B)
        pos = (self.da_gain * reward.clamp(min=0)).clamp(max=3.0)
        neg = (self.da_gain * (-reward).clamp(min=0)).clamp(max=3.0)
        if len(self.brain.idx_dan_reward):
            I[self.brain.idx_dan_reward] += pos
        if len(self.brain.idx_dan_punish):
            I[self.brain.idx_dan_punish] += neg
        return I

    @torch.no_grad()
    def update(self, r: torch.Tensor) -> torch.Tensor:
        """Apply the three-factor rule after a brain step. Returns the dopamine error per column."""
        c = self.cfg
        r_post = r.index_select(0, self.post_neurons)
        self.post_avg += (r_post - self.post_avg) / c.post_avg_tau
        dev = (r_post - self.post_avg).index_select(0, self.post_local)
        self.elig.mul_(self.lam).add_(r.index_select(0, self.pre) * dev)
        da = self.brain.dopamine(r)
        err = da - self.da_avg
        self.da_avg += err / c.da_avg_tau
        self.dW.mul_(1.0 - c.decay).add_(self.elig * (self.eta * err))
        torch.maximum(self.dW, -self.w0[:, None], out=self.dW)
        torch.minimum(self.dW, self.w0[:, None], out=self.dW)
        return err

    def state_dict(self) -> dict:
        return {"cfg": self.cfg.to_dict(), "dW": self.dW.clone(), "eta": self.eta, "lam": self.lam,
                "da_gain": self.da_gain}
