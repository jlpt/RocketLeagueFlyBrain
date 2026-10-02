"""Stage 2: evolution strategies with an indirect (cell-type) encoding.

The genome does not list synapses or neurons. It lists *cell types*: one gene
for the presynaptic gain and one for the postsynaptic gain of every annotated
cell type (rare types share a gene with their superclass/class), plus the
plasticity genes (learning rate, eligibility decay, reward->dopamine gain) and
a motor bias. A gene therefore changes every neuron of a type at once -- the
way development and neuromodulation act on real brains -- and the search
space shrinks from ~4e5 per-neuron parameters to a few thousand.

Why evolution here (and not for stage 1): after imitation the brain already
plays; evolution can now optimise what imitation cannot see -- the actual
match outcome (goals, touches, shots) -- needs no gradient through the game,
handles the non-differentiable plasticity, and runs a whole population in a
single sparse matrix product (each member is just a different set of gains).

Algorithm: OpenAI-ES (Salimans et al. 2017) with antithetic sampling,
common random numbers per pair, centred-rank fitness shaping and Adam.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..brain.controller import FlyController
from ..brain.model import FlyBrain, save_brain
from ..brain.plasticity import DopaminePlasticity, PlasticityConfig
from ..connectome.graph import BrainGraph
from ..sim.agents import BallChaseAgent, TeacherAgent
from ..sim.env import Match
from .rollout import fly_matches

N_GENES_PLASTIC = 3
N_MOTOR_BIAS = 8


@dataclass
class ESConfig:
    pairs: int = 16                 # population = 2 * pairs
    sigma: float = 0.05
    lr: float = 0.02
    generations: int = 40
    episode_seconds: float = 45.0
    min_type_size: int = 4
    plastic: int = 1
    opponent: str = "teacher"
    seed: int = 0


def es_units(graph: BrainGraph, min_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Map every neuron to an indirect-encoding unit (cell type or coarse fallback)."""
    counts = np.bincount(graph.type_id, minlength=graph.n_types)
    big = counts[graph.type_id] >= min_size
    fine = graph.type_names[graph.type_id].astype("U")
    coarse = np.char.add(np.char.add(graph.superclass.astype("U"), "/"), graph.cell_class.astype("U"))
    names = np.where(big, fine, coarse)
    unit_names, unit_id = np.unique(names, return_inverse=True)
    return unit_names, unit_id.astype(np.int64)


class Genome:
    def __init__(self, n_units: int, pcfg: PlasticityConfig):
        self.U = n_units
        self.size = 2 * n_units + N_GENES_PLASTIC + N_MOTOR_BIAS
        self.p0 = np.array([np.log(pcfg.eta), np.log(pcfg.trace_lambda / (1 - pcfg.trace_lambda)),
                            np.log(pcfg.da_gain)], np.float32)

    def split(self, x: np.ndarray):
        U = self.U
        pre, post = x[..., :U], x[..., U:2 * U]
        pg = x[..., 2 * U:2 * U + N_GENES_PLASTIC] + self.p0
        mb = x[..., 2 * U + N_GENES_PLASTIC:]
        return pre, post, pg, mb


def apply_population(brain: FlyBrain, ctrl: FlyController, genome: Genome, unit_id: torch.Tensor,
                     pop: np.ndarray) -> None:
    pre, post, pg, mb = genome.split(pop)                 # (P, U) ...
    gp, gq = brain.gains()
    with torch.no_grad():
        off_pre = torch.from_numpy(pre.T.astype(np.float32))[unit_id]     # (N, P)
        off_post = torch.from_numpy(post.T.astype(np.float32))[unit_id]
        ctrl.g_pre = gp[:, None] * torch.exp(off_pre)
        ctrl.g_post = gq[:, None] * torch.exp(off_post)
    ctrl.logit_bias = torch.from_numpy(mb.astype(np.float32))
    if ctrl.plasticity is not None:
        ctrl.plasticity.set_genes(eta=np.exp(pg[:, 0]), lam=1 / (1 + np.exp(-pg[:, 1])), da_gain=np.exp(pg[:, 2]))


def fold_genome(brain: FlyBrain, genome: Genome, unit_id: torch.Tensor, mu: np.ndarray) -> dict:
    """Bake the evolved mean genome into the brain; return the evolved plasticity genes."""
    pre, post, pg, mb = genome.split(mu)
    with torch.no_grad():
        brain.theta_pre += torch.from_numpy(pre.astype(np.float32))[unit_id]
        brain.theta_post += torch.from_numpy(post.astype(np.float32))[unit_id]
        brain.readout.bias += torch.from_numpy(mb.astype(np.float32))
    return {"eta": float(np.exp(pg[0])), "trace_lambda": float(1 / (1 + np.exp(-pg[1]))),
            "da_gain": float(np.exp(pg[2]))}


def _centered_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999):
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0
        self.lr, self.b1, self.b2 = lr, b1, b2

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + 1e-8)


def run_es(brain: FlyBrain, graph: BrainGraph, cfg: ESConfig, out_dir: str | Path, resume: bool = False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    unit_names, unit_np = es_units(graph, cfg.min_type_size)
    unit_id = torch.from_numpy(unit_np)
    pcfg = PlasticityConfig()
    genome = Genome(len(unit_names), pcfg)
    print(f"[es] {graph.n} neurons -> {len(unit_names)} cell-type units; genome size {genome.size}", flush=True)
    P = 2 * cfg.pairs
    plast = DopaminePlasticity(brain, graph, pcfg) if cfg.plastic else None
    ctrl = FlyController(brain, P, plasticity=plast)
    mu = np.zeros(genome.size)
    opt = Adam(genome.size, cfg.lr)
    start = 0
    state_path = out / "es_state.npz"
    if resume and state_path.exists():
        z = np.load(state_path)
        mu, opt.m, opt.v, opt.t, start = z["mu"], z["m"], z["v"], int(z["t"]), int(z["gen"]) + 1
    rng = np.random.default_rng(cfg.seed + start)
    n_steps = int(cfg.episode_seconds * 120 / brain.cfg.tick_skip)
    for gen in range(start, cfg.generations):
        t0 = time.time()
        eps = rng.standard_normal((cfg.pairs, genome.size))
        pop = np.concatenate([mu + cfg.sigma * eps, mu - cfg.sigma * eps])
        ctrl.reset()                                      # fresh brains (and plastic weights) per generation
        apply_population(brain, ctrl, genome, unit_id, pop)
        seeds = rng.integers(1 << 30, size=cfg.pairs)
        seeds = np.concatenate([seeds, seeds])            # common random numbers per antithetic pair
        matches = [Match(seed=int(s), max_seconds=1e9, kickoff_prob=0.5) for s in seeds]
        opps = [TeacherAgent(seed=int(s)) if cfg.opponent == "teacher" else BallChaseAgent() for s in seeds]
        res = fly_matches(ctrl, matches, opps, n_steps, dopamine=plast is not None, first_reset="mixed",
                          after_goal="kickoff")
        f = res["fitness"]
        u = _centered_ranks(f)
        grad = ((u[:cfg.pairs] - u[cfg.pairs:])[:, None] * eps).sum(0) / (2 * cfg.pairs * cfg.sigma)
        mu = mu + opt.step(grad)
        log = {"gen": gen, "fit_mean": round(float(f.mean()), 3), "fit_max": round(float(f.max()), 3),
               "goals_for": int(res["goals_for"].sum()), "goals_against": int(res["goals_against"].sum()),
               "touches_per_min": round(float(res["touches"].sum()) / (P * cfg.episode_seconds / 60), 2),
               "mu_norm": round(float(np.linalg.norm(mu)), 3), "secs": round(time.time() - t0, 1)}
        print(json.dumps(log), flush=True)
        with open(out / "es_log.jsonl", "a") as fh:
            fh.write(json.dumps(log) + "\n")
        np.savez(state_path, mu=mu, m=opt.m, v=opt.v, t=opt.t, gen=gen)

    genes = fold_genome(brain, genome, unit_id, mu)
    pc = asdict(pcfg)
    pc.update(genes)
    pc["sites"] = list(pc["sites"])
    save_brain(out / "es_final.pt", brain, {"stage": "es", "plasticity": pc, "es_cfg": asdict(cfg),
                                            "n_units": int(len(unit_names))})
    print(f"[es] saved {out / 'es_final.pt'}  plasticity genes: {genes}", flush=True)
    return brain
