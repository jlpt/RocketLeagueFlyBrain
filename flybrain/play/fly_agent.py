"""A single fly brain as a drop-in agent (RocketSim matches, RLBot, fly-vs-fly)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..brain.controller import FlyController
from ..brain.model import FlyBrain, load_brain
from ..connectome.graph import BrainGraph
from ..eval.evaluate import plasticity_from_extra
from ..sim.agents import Agent
from ..sim.obs import build_obs
from ..sim.state import View


class FlyAgent(Agent):
    name = "fly"

    def __init__(self, brain: FlyBrain, graph: BrainGraph | None = None, extra: dict | None = None,
                 plastic: bool = False):
        plast = plasticity_from_extra(brain, graph, extra or {}) if plastic else None
        self.ctrl = FlyController(brain, 1, plasticity=plast)
        self.prev = np.zeros(8, np.float32)

    @classmethod
    def from_files(cls, graph_path: str | Path, checkpoint: str | Path, plastic: bool = False) -> "FlyAgent":
        graph = BrainGraph.load(graph_path)
        brain, extra = load_brain(checkpoint, graph)
        brain.eval()
        return cls(brain, graph, extra, plastic)

    def reset(self) -> None:
        """New kickoff: clear neural activity and delay lines (learned plastic weights are kept)."""
        self.ctrl.reset_columns([0])
        self.prev[:] = 0

    def act(self, view: View, match=None, reward: float | None = None) -> np.ndarray:
        obs = build_obs(view, self.prev)[None]
        r = None if reward is None else np.array([reward], np.float32)
        a = self.ctrl.step(obs, r)[0]
        self.prev = a.astype(np.float32).copy()
        return a
