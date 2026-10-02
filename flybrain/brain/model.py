"""The fly brain as a recurrent controller.

Every neuron of the (central brain + nerve cord) connectome is a leaky rate
unit. The wiring, the synapse counts and the synapse signs come from the male
CNS connectome and are never changed by training. What is learned:

  * per-neuron presynaptic gain  ``g_pre``  (release strength)
  * per-neuron postsynaptic gain ``g_post`` (dendritic gain / excitability)
  * per-neuron bias and membrane time constant
  * the sensory transduction (observation -> sensory neuron currents)
  * the motor transduction (descending + motor neuron rates -> car controls)

so the effective synapse is ``w_ij = g_post_i * g_pre_j * sign_j * log1p(n_ij) / S_i``
(Dale's law preserved, connectome structure preserved). On top of that,
``flybrain.brain.plasticity`` adds dopamine-gated synaptic plasticity.

Timing (human-like reaction time)
---------------------------------
A real fly reacts to a looming object in ~30-50 ms; a human needs ~200-250 ms.
We time-dilate the fly: one brain step = one synaptic hop = 33.3 ms of game
time (4 physics ticks at 120 Hz), i.e. propagation is ~10x slower than in the
fly. On top of that come two fixed delays that are not simulated as neurons:

  * sensory delay (default 3 steps = 100 ms): phototransduction + optic lobe
    processing (the optic lobe is replaced by the visual feature encoder), and
  * motor delay   (default 1 step  =  33 ms): motor neuron -> muscle -> input.

The shortest sensory->descending neuron path is 1 hop, so the hard floor is
(3 + 1 + 1) * 33 ms = 167 ms; leaky integration and multi-hop paths put the
measured reaction time at ~200-250 ms (see ``flybrain.brain.reaction``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from ..connectome.graph import BrainGraph
from ..sim.obs import GROUP_SLICES, OBS_DIM

N_ACTIONS = 8
N_ANALOG = 5


@dataclass
class BrainConfig:
    tick_skip: int = 4                # physics ticks per brain step (120 Hz physics)
    sensory_delay: int = 3            # brain steps
    motor_delay: int = 1              # brain steps
    tau_init_ms: float = 50.0
    tau_min_ms: float = 34.0          # >= one step, so a signal crosses at most one synapse per step
    tau_max_ms: float = 500.0
    g_post_init: float = 4.0
    bias_init: float = -0.05
    in_scale: float = 1.0
    readout: str = "dn+mn"            # which neurons drive the car

    @property
    def step_ms(self) -> float:
        return 1000.0 * self.tick_skip / 120.0

    def to_dict(self) -> dict:
        return asdict(self)


class _SpMM(torch.autograd.Function):
    """y = W @ x for a fixed CSR matrix W, with dL/dx = W^T @ dL/dy."""

    @staticmethod
    def forward(ctx, x, W, WT):
        ctx.WT = WT
        return torch.sparse.mm(W, x) if W.layout == torch.sparse_coo else W @ x

    @staticmethod
    def backward(ctx, g):
        return ctx.WT @ g.contiguous(), None, None


def spmm(W: torch.Tensor, WT: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _SpMM.apply(x, W, WT)


class FlyBrain(nn.Module):
    def __init__(self, graph: BrainGraph, cfg: BrainConfig | None = None, seed: int = 0):
        super().__init__()
        self.cfg = cfg = cfg or BrainConfig()
        self.graph_meta = dict(graph.meta)
        n = graph.n
        self.n = n
        gen = torch.Generator().manual_seed(seed)

        # -------- fixed connectome (normalised signed synapse strengths)
        mag = np.log1p(graph.syn.astype(np.float32))
        row_sum = np.add.reduceat(mag, graph.indptr[:-1]) if graph.nnz else np.zeros(n, np.float32)
        row_sum = np.where(np.diff(graph.indptr) > 0, row_sum, 1.0).astype(np.float32)
        post = graph.post_index()
        w = graph.sign[graph.indices] * mag / row_sum[post]
        # derived from the graph -> not stored in checkpoints (keeps them ~2 MB)
        self.register_buffer("w_values", torch.from_numpy(w.astype(np.float32)), persistent=False)
        self.register_buffer("crow", torch.from_numpy(graph.indptr.astype(np.int64)), persistent=False)
        self.register_buffer("col", torch.from_numpy(graph.indices.astype(np.int64)), persistent=False)
        self._build_sparse()

        # -------- neuron groups
        self.in_groups = [g for g in GROUP_SLICES if g in graph.groups and len(graph.groups[g])]
        for g in self.in_groups:
            self.register_buffer(f"idx_{g}", torch.from_numpy(graph.groups[g].astype(np.int64)), persistent=False)
        parts = []
        if "dn" in cfg.readout:
            parts.append(graph.groups["out_dn"])
        if "mn" in cfg.readout:
            parts.append(graph.groups["out_mn"])
        out_idx = np.unique(np.concatenate(parts)).astype(np.int64)
        self.register_buffer("idx_out", torch.from_numpy(out_idx), persistent=False)
        for g in ("dan_reward", "dan_punish"):
            arr = graph.groups.get(g, np.zeros(0, np.int64)).astype(np.int64)
            self.register_buffer(f"idx_{g}", torch.from_numpy(arr), persistent=False)

        # -------- learnable parameters
        self.theta_pre = nn.Parameter(torch.zeros(n))
        self.theta_post = nn.Parameter(torch.zeros(n))
        self.bias = nn.Parameter(torch.full((n,), cfg.bias_init))
        p0 = (cfg.tau_init_ms - cfg.tau_min_ms) / (cfg.tau_max_ms - cfg.tau_min_ms)
        self.tau_raw = nn.Parameter(torch.full((n,), float(np.log(p0 / (1 - p0)))))
        self.w_in = nn.ParameterDict()
        for g in self.in_groups:
            sl = GROUP_SLICES[g]
            d = sl.stop - sl.start
            k = len(graph.groups[g])
            self.w_in[g] = nn.Parameter(torch.randn(k, d, generator=gen) * (cfg.in_scale / np.sqrt(d)))
        n_out = len(out_idx)
        self.readout = nn.Linear(n_out, N_ACTIONS)
        with torch.no_grad():
            self.readout.weight.copy_(torch.randn(N_ACTIONS, n_out, generator=gen) * (1.0 / np.sqrt(n_out)))
            self.readout.bias.zero_()

    # ------------------------------------------------------------- helpers
    def _build_sparse(self):
        n = self.n
        W = torch.sparse_csr_tensor(self.crow, self.col, self.w_values, size=(n, n))
        WT = W.to_sparse_coo().t().coalesce().to_sparse_csr()
        self._W, self._WT = W, WT

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        self._build_sparse()
        return out

    def load_state_dict(self, *args, **kwargs):
        res = super().load_state_dict(*args, **kwargs)
        self._build_sparse()
        return res

    def gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.exp(self.theta_pre), self.cfg.g_post_init * torch.exp(self.theta_post)

    def alpha(self) -> torch.Tensor:
        c = self.cfg
        tau = c.tau_min_ms + (c.tau_max_ms - c.tau_min_ms) * torch.sigmoid(self.tau_raw)
        return c.step_ms / tau          # <= 1

    def init_state(self, batch: int) -> dict:
        z = torch.zeros(self.n, batch, device=self.w_values.device)
        return {"v": z, "r": z.clone()}

    def sensory_current(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, OBS_DIM) -> (N, B) current injected into sensory neurons."""
        B = obs.shape[0]
        I = torch.zeros(self.n, B, device=obs.device, dtype=obs.dtype)
        for g in self.in_groups:
            sl = GROUP_SLICES[g]
            idx = getattr(self, f"idx_{g}")
            I = I.index_add(0, idx, self.w_in[g] @ obs[:, sl].T)
        return I

    # ---------------------------------------------------------------- step
    def step(self, state: dict, obs: torch.Tensor, extra_current: torch.Tensor | None = None,
             g_pre: torch.Tensor | None = None, g_post: torch.Tensor | None = None,
             plastic=None, noise_std: float = 0.0) -> tuple[dict, torch.Tensor]:
        """Advance one synaptic hop.

        obs: (B, OBS_DIM) already sensory-delayed observation.
        g_pre/g_post: optional (N,1) or (N,B) gains (ES population members).
        plastic: optional plasticity module providing ``current(r)``.
        Returns new state and action logits (B, 8).
        """
        if g_pre is None or g_post is None:
            gp, gq = self.gains()
            g_pre = gp[:, None] if g_pre is None else g_pre
            g_post = gq[:, None] if g_post is None else g_post
        r = state["r"]
        syn = spmm(self._W, self._WT, g_pre * r) * g_post
        if plastic is not None:
            syn = syn + plastic.current(r, g_pre, g_post)
        drive = syn + self.bias[:, None] + self.sensory_current(obs)
        if extra_current is not None:
            drive = drive + extra_current
        if noise_std > 0:
            drive = drive + noise_std * torch.randn_like(drive)
        a = self.alpha()[:, None]
        v = state["v"] + a * (drive - state["v"])
        r_new = torch.tanh(torch.relu(v))
        logits = self.readout(r_new.index_select(0, self.idx_out).T)
        return {"v": v, "r": r_new}, logits

    @staticmethod
    def logits_to_action(logits: torch.Tensor, sample: bool = False) -> torch.Tensor:
        analog = torch.tanh(logits[..., :N_ANALOG])
        p = torch.sigmoid(logits[..., N_ANALOG:])
        buttons = torch.bernoulli(p) if sample else (p > 0.5).float()
        return torch.cat([analog, buttons], dim=-1)

    def dopamine(self, r: torch.Tensor) -> torch.Tensor:
        """Net dopamine signal per batch column: reward DANs (PAM) minus punishment DANs (PPL1)."""
        da = torch.zeros(r.shape[1], device=r.device)
        if len(self.idx_dan_reward):
            da = da + r.index_select(0, self.idx_dan_reward).mean(0)
        if len(self.idx_dan_punish):
            da = da - r.index_select(0, self.idx_dan_punish).mean(0)
        return da

    def param_groups(self, lr: float) -> list[dict]:
        neuron = [self.theta_pre, self.theta_post, self.bias, self.tau_raw]
        io = list(self.w_in.parameters()) + list(self.readout.parameters())
        return [{"params": neuron, "lr": lr}, {"params": io, "lr": lr}]


class DelayLine:
    """Fixed-length FIFO used for the sensory and motor delays. Batch is the first dim."""

    def __init__(self, length: int, shape: tuple):
        self.length = length
        self.buf = [torch.zeros(shape) for _ in range(length)]

    def push(self, x: torch.Tensor) -> torch.Tensor:
        """Insert x, return the element from ``length`` steps ago (x itself if length == 0)."""
        if self.length == 0:
            return x
        out = self.buf.pop(0)
        self.buf.append(x.clone())
        return out

    def reset_rows(self, rows) -> None:
        for b in self.buf:
            b[rows] = 0.0


def save_brain(path, brain: FlyBrain, extra: dict | None = None) -> None:
    torch.save({"cfg": brain.cfg.to_dict(), "state_dict": brain.state_dict(),
                "graph_meta": brain.graph_meta, "extra": extra or {}}, path)


def load_brain(path, graph: BrainGraph) -> tuple[FlyBrain, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    brain = FlyBrain(graph, BrainConfig(**ck["cfg"]))
    brain.load_state_dict(ck["state_dict"])
    return brain, ck.get("extra", {})
