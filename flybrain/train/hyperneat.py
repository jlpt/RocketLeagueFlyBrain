"""HyperNEAT-style evolution: a CPPN paints synapse strengths onto the fly connectome.

The substrate is the real fly brain: every neuron has a 3D soma position
(left/right, dorsal/ventral, head->nerve cord), a side, a sign (excitatory or
inhibitory) and a sensorimotor "layer". For every one of the 3.46M existing
synapses the CPPN is queried with the coordinates of its two neurons

    CPPN(|u|_pre, v_pre, w_pre, side_pre, sign_pre, layer_pre,
         |u|_post, ..., layer_post, same_side, distance)  ->  o

and the synapse is scaled by exp(GAIN_SCALE * o). Which synapses exist and
their signs stay fixed (the connectome), so this is HyperNEAT on a fixed
substrate: the CPPN only sets strengths, smoothly in space, with mirror
symmetry available through |u| and same_side. NEAT (neat-python) evolves the
CPPN's weights *and topology*, starting from tiny networks.

Everything starts from the imitation-trained brain (o = 0 -> unchanged), the
whole population runs as one block-diagonal sparse product, and the final
champion is chosen by re-evaluating the best genomes against the unmodified
brain on held-out matches, so a lucky single game can't win.
"""

from __future__ import annotations

import copy
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import neat
import numpy as np
import torch
from neat.graphs import feed_forward_layers

from ..brain.controller import FlyController
from ..brain.model import FlyBrain, save_brain
from ..connectome.graph import BrainGraph
from ..connectome.positions import neuron_features
from ..sim.agents import TeacherAgent
from ..sim.env import Match
from .rollout import fly_matches

GAIN_SCALE = 0.5
N_INPUTS = 14

NEAT_CONFIG = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1e9
no_fitness_termination = True
pop_size              = {pop}
reset_on_extinction   = True

[DefaultGenome]
num_inputs              = {n_in}
num_hidden              = 0
num_outputs             = 1
initial_connection      = partial_direct 0.2
feed_forward            = True
activation_default      = tanh
activation_mutate_rate  = 0.2
activation_options      = tanh sin gauss sigmoid abs identity
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 0.05
bias_max_value          = 3.0
bias_min_value          = -3.0
bias_mutate_power       = 0.1
bias_mutate_rate        = 0.5
bias_replace_rate       = 0.02
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 3.0
response_min_value      = -3.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 0.2
weight_max_value        = 3.0
weight_min_value        = -3.0
weight_mutate_power     = 0.15
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.05
enabled_default         = True
enabled_mutate_rate     = 0.02
conn_add_prob           = 0.3
conn_delete_prob        = 0.1
node_add_prob           = 0.15
node_delete_prob        = 0.05
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
single_structural_mutation = False

[DefaultSpeciesSet]
compatibility_threshold = 2.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 8
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.3
min_species_size   = 2
"""


@dataclass
class HyperNEATConfig:
    pop: int = 24
    generations: int = 20
    episode_seconds: float = 40.0
    final_candidates: int = 6
    final_seeds: int = 8
    final_seconds: float = 60.0
    seed: int = 0


# numpy versions of neat-python's activation functions (same scaling)
ACT = {
    "tanh": lambda z: np.tanh(np.clip(2.5 * z, -60, 60)),
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-np.clip(5.0 * z, -60, 60))),
    "sin": lambda z: np.sin(np.clip(5.0 * z, -60, 60)),
    "gauss": lambda z: np.exp(-5.0 * np.clip(z, -3.4, 3.4) ** 2),
    "abs": np.abs,
    "identity": lambda z: z,
    "relu": lambda z: np.maximum(z, 0.0),
    "clamped": lambda z: np.clip(z, -1.0, 1.0),
}


def make_config(path: Path, pop: int) -> neat.Config:
    path.write_text(NEAT_CONFIG.format(pop=pop, n_in=N_INPUTS))
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(path))


def edge_inputs(graph: BrainGraph) -> np.ndarray:
    """(E, 14) CPPN query for every synapse: pre features, post features, same side, distance."""
    f = neuron_features(graph)
    pre = graph.indices.astype(np.int64)
    post = graph.post_index().astype(np.int64)
    same = (f[pre, 3] * f[post, 3])[:, None]
    dist = np.linalg.norm(graph.pos[pre] - graph.pos[post], axis=1)[:, None]
    return np.concatenate([f[pre], f[post], same, dist], axis=1).astype(np.float32)


def cppn_eval(genome, config: neat.Config, X: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of neat.nn.FeedForwardNetwork over all rows of X."""
    gc = config.genome_config
    conns = [cg.key for cg in genome.connections.values() if cg.enabled]
    layers, _ = feed_forward_layers(gc.input_keys, gc.output_keys, conns)
    values = {k: X[:, i] for i, k in enumerate(gc.input_keys)}
    incoming: dict[int, list] = {}
    for (i, o) in conns:
        incoming.setdefault(o, []).append((i, genome.connections[(i, o)].weight))
    for layer in layers:
        for node in layer:
            s = np.zeros(X.shape[0], np.float32)
            for i, w in incoming.get(node, []):
                if i in values:
                    s += np.float32(w) * values[i]
            ng = genome.nodes[node]
            values[node] = ACT[ng.activation](np.float32(ng.bias) + np.float32(ng.response) * s).astype(np.float32)
    out = gc.output_keys[0]
    return values.get(out, np.zeros(X.shape[0], np.float32))


def genome_gain(genome, config, X) -> np.ndarray:
    return np.exp(GAIN_SCALE * np.clip(cppn_eval(genome, config, X), -2.0, 2.0)).astype(np.float32)


class PopulationSynapses:
    """P copies of the connectome with different synapse strengths, as one block-diagonal CSR."""

    def __init__(self, brain: FlyBrain, P: int):
        N, E = brain.n, brain.col.numel()
        self.N, self.E, self.P = N, E, P
        self.crow = torch.cat([brain.crow[:-1] + p * E for p in range(P)] + [torch.tensor([P * E])]).int()
        self.col = torch.cat([brain.col + p * N for p in range(P)]).int()
        self.w = brain.w_values.clone()
        self.W = None

    def set_gains(self, gains: np.ndarray) -> None:
        vals = (self.w[None, :] * torch.from_numpy(gains)).reshape(-1)
        self.W = torch.sparse_csr_tensor(self.crow, self.col, vals, size=(self.P * self.N, self.P * self.N))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W @ x.T.contiguous().reshape(-1, 1)
        return y.reshape(self.P, self.N).T


def evaluate_gains(brain: FlyBrain, gains: np.ndarray, seeds: list[int], seconds: float) -> dict:
    """Column b plays one match vs the teacher with synapse gains[b] and match seed seeds[b]."""
    P = len(gains)
    syn = PopulationSynapses(brain, P)
    syn.set_gains(np.ascontiguousarray(gains))
    ctrl = FlyController(brain, P)
    ctrl.synapses = syn
    matches = [Match(seed=int(s), max_seconds=1e9, kickoff_prob=0.5) for s in seeds]
    opps = [TeacherAgent(seed=int(s)) for s in seeds]
    n = int(seconds * 120 / brain.cfg.tick_skip)
    return fly_matches(ctrl, matches, opps, n, dopamine=False, first_reset="mixed")


def run_hyperneat(brain: FlyBrain, graph: BrainGraph, cfg: HyperNEATConfig, out_dir: str | Path,
                  resume: bool = True) -> FlyBrain:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    brain.eval()
    X = edge_inputs(graph)
    config = make_config(out / "neat.cfg", cfg.pop)
    ckpts = sorted(out.glob("neat-ckpt-*"), key=lambda p: int(p.name.split("-")[-1]))
    if resume and ckpts:
        pop = neat.Checkpointer.restore_checkpoint(str(ckpts[-1]))
        print(f"[hyperneat] resumed from {ckpts[-1].name}", flush=True)
    else:
        pop = neat.Population(config, seed=cfg.seed)
    pop.add_reporter(neat.Checkpointer(generation_interval=1, time_interval_seconds=None,
                                       filename_prefix=str(out / "neat-ckpt-")))
    hof_path = out / "hall_of_fame.pkl"
    hall = pickle.load(open(hof_path, "rb")) if (resume and hof_path.exists()) else []
    rng = np.random.default_rng(cfg.seed + pop.generation)

    def eval_genomes(genomes, config):
        t0 = time.time()
        gains = np.stack([genome_gain(g, config, X) for _, g in genomes])
        seed = int(rng.integers(1 << 30))
        res = evaluate_gains(brain, gains, [seed] * len(genomes), cfg.episode_seconds)
        for (gid, g), f in zip(genomes, res["fitness"]):
            g.fitness = float(f)
        order = np.argsort(-res["fitness"])
        for k in order[:2]:
            hall.append((float(res["fitness"][k]), copy.deepcopy(genomes[k][1])))
        pickle.dump(hall, open(hof_path, "wb"))
        log = {"gen": pop.generation, "fit_mean": round(float(res["fitness"].mean()), 3),
               "fit_max": round(float(res["fitness"].max()), 3),
               "goals_for": int(res["goals_for"].sum()), "goals_against": int(res["goals_against"].sum()),
               "touches_per_min": round(float(res["touches"].sum()) / (len(genomes) * cfg.episode_seconds / 60), 2),
               "species": len(pop.species.species) if pop.species else 0,
               "cppn_nodes_best": len(genomes[order[0]][1].nodes),
               "gain_range_best": [round(float(gains[order[0]].min()), 3), round(float(gains[order[0]].max()), 3)],
               "secs": round(time.time() - t0, 1)}
        print(json.dumps(log), flush=True)
        with open(out / "hyperneat_log.jsonl", "a") as fh:
            fh.write(json.dumps(log) + "\n")

    remaining = cfg.generations - pop.generation
    if remaining > 0:
        pop.run(eval_genomes, remaining)

    # ---- champion selection on held-out matches (the unmodified brain competes too)
    hall.sort(key=lambda t: -t[0])
    cands = [g for _, g in hall[:cfg.final_candidates]]
    cand_gains = [np.ones(len(X), np.float32)] + [genome_gain(g, config, X) for g in cands]
    seeds = [10_000 + i for i in range(cfg.final_seeds)]
    k = len(cand_gains)
    scores = np.zeros((k, 3))
    for s_chunk in range(0, len(seeds), max(1, 28 // k)):
        ss = seeds[s_chunk:s_chunk + max(1, 28 // k)]
        gains = np.stack([cand_gains[c] for c in range(k) for _ in ss])
        res = evaluate_gains(brain, gains, [s for _ in range(k) for s in ss], cfg.final_seconds)
        for c in range(k):
            sl = slice(c * len(ss), (c + 1) * len(ss))
            scores[c] += [res["fitness"][sl].sum(), res["goals_for"][sl].sum(), res["goals_against"][sl].sum()]
    scores[:, 0] /= len(seeds)
    best = int(np.argmax(scores[:, 0]))
    report = {"candidates": [{"name": "imitation brain (no CPPN)" if c == 0 else f"hall-of-fame #{c}",
                              "mean_fitness": round(float(scores[c, 0]), 3), "goals_for": int(scores[c, 1]),
                              "goals_against": int(scores[c, 2])} for c in range(k)],
              "winner": best, "seeds": len(seeds), "seconds": cfg.final_seconds}
    print(json.dumps(report), flush=True)
    (out / "champion_report.json").write_text(json.dumps(report, indent=2))
    if best > 0:
        brain.set_edge_gain(cand_gains[best])
        pickle.dump(cands[best - 1], open(out / "champion_cppn.pkl", "wb"))
    save_brain(out / "hyperneat_final.pt", brain, {"stage": "hyperneat", "report": report,
                                                   "hyperneat_cfg": asdict(cfg)})
    return brain
