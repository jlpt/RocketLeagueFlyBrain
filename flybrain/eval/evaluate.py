"""Evaluate a fly brain in full 1v1 matches against scripted opponents."""

from __future__ import annotations

import time

import numpy as np

from ..brain.controller import FlyController
from ..brain.model import FlyBrain
from ..brain.plasticity import DopaminePlasticity, PlasticityConfig
from ..brain.reaction import measure_reaction_time
from ..connectome.graph import BrainGraph
from ..sim.agents import BallChaseAgent, IdleAgent, RandomAgent, TeacherAgent, play
from ..sim.env import Match
from ..train.rollout import fly_matches

OPPONENTS = {
    "teacher": lambda s: TeacherAgent(seed=s),
    "ballchase": lambda s: BallChaseAgent(),
    "idle": lambda s: IdleAgent(),
    "random": lambda s: RandomAgent(seed=s),
}


def plasticity_from_extra(brain: FlyBrain, graph: BrainGraph, extra: dict) -> DopaminePlasticity:
    pc = dict(extra.get("plasticity", {}))
    if "sites" in pc:
        pc["sites"] = tuple(pc["sites"])
    return DopaminePlasticity(brain, graph, PlasticityConfig(**pc))


def _summary(gf: np.ndarray, ga: np.ndarray, touches: np.ndarray, seconds: float) -> dict:
    return {
        "goals_for": int(gf.sum()), "goals_against": int(ga.sum()),
        "wins": int((gf > ga).sum()), "draws": int((gf == ga).sum()), "losses": int((gf < ga).sum()),
        "goal_diff_per_5min": round(float((gf - ga).mean()) * 300.0 / seconds, 2),
        "touches_per_min": round(float(touches.mean()) * 60.0 / seconds, 2),
    }


def evaluate(brain: FlyBrain, graph: BrainGraph, extra: dict, opponents=("teacher", "ballchase", "idle"),
             seconds: float = 300.0, n_matches: int = 4, plastic: bool = False, seed: int = 123,
             baseline: bool = True) -> dict:
    """Fly (blue) vs each opponent: n_matches parallel matches of ``seconds`` with kickoffs after goals.
    Also reports the teacher bot against the same opponents as a reference point."""
    brain.eval()
    results = {"seconds_per_match": seconds, "matches": n_matches, "plastic": plastic}
    n_steps = int(seconds * 120 / brain.cfg.tick_skip)
    for name in opponents:
        t0 = time.time()
        plast = plasticity_from_extra(brain, graph, extra) if plastic else None
        ctrl = FlyController(brain, n_matches, plasticity=plast)
        matches = [Match(seed=seed + i, max_seconds=1e9) for i in range(n_matches)]
        opps = [OPPONENTS[name](seed + i) for i in range(n_matches)]
        res = fly_matches(ctrl, matches, opps, n_steps, dopamine=plastic)
        r = {"fly": _summary(res["goals_for"], res["goals_against"], res["touches"], seconds)}
        if baseline and name != "teacher":
            gf, ga, tc = [], [], []
            for i in range(n_matches):
                p = play(TeacherAgent(seed=seed + i), OPPONENTS[name](seed + i), seconds=seconds, seed=seed + i)
                gf.append(p["score"][0]); ga.append(p["score"][1]); tc.append(p["touches"][0])
            r["teacher_bot"] = _summary(np.array(gf), np.array(ga), np.array(tc), seconds)
        r["eval_s"] = round(time.time() - t0, 1)
        results[f"vs_{name}"] = r
    results["reaction_time"] = measure_reaction_time(brain, n_pairs=64)
    return results
