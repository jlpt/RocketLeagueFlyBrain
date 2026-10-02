"""Runs a batch of fly brains in lock-step with the game, including the delays."""

from __future__ import annotations

import numpy as np
import torch

from ..sim.obs import OBS_DIM
from .model import N_ACTIONS, DelayLine, FlyBrain
from .plasticity import DopaminePlasticity


class FlyController:
    """B independent flies (one column each) sharing the same connectome.

    Per brain step: observation -> sensory delay -> one synaptic hop ->
    readout -> motor delay -> car controls.
    """

    def __init__(self, brain: FlyBrain, batch: int, plasticity: DopaminePlasticity | None = None,
                 noise_std: float = 0.0, sample_buttons: bool = False):
        self.brain = brain
        self.B = batch
        self.plasticity = plasticity
        self.noise_std = noise_std
        self.sample_buttons = sample_buttons
        self.g_pre = None      # optional (N, B) gains (evolution members)
        self.g_post = None
        self.logit_bias = None  # optional (B, 8) motor bias (evolution members)
        self.reset()

    def reset(self) -> None:
        c = self.brain.cfg
        self.state = self.brain.init_state(self.B)
        self.obs_delay = DelayLine(c.sensory_delay, (self.B, OBS_DIM))
        self.act_delay = DelayLine(c.motor_delay, (self.B, N_ACTIONS))
        if self.plasticity is not None:
            self.plasticity.reset(self.B)

    def reset_columns(self, cols) -> None:
        """Start a fresh episode for some flies (state & delay lines; learned dW persists)."""
        for k in self.state:
            self.state[k][:, cols] = 0.0
        self.obs_delay.reset_rows(cols)
        self.act_delay.reset_rows(cols)

    @torch.no_grad()
    def step(self, obs: np.ndarray, reward: np.ndarray | None = None) -> np.ndarray:
        """obs: (B, OBS_DIM) current observations. Returns the (B, 8) controls applied now."""
        o = self.obs_delay.push(torch.as_tensor(obs, dtype=torch.float32))
        extra = None
        if self.plasticity is not None and reward is not None:
            extra = self.plasticity.reward_current(torch.as_tensor(reward, dtype=torch.float32))
        self.state, logits = self.brain.step(self.state, o, extra, self.g_pre, self.g_post,
                                             plastic=self.plasticity, noise_std=self.noise_std)
        if self.plasticity is not None:
            self.plasticity.update(self.state["r"])
        if self.logit_bias is not None:
            logits = logits + self.logit_bias
        a = FlyBrain.logits_to_action(logits, sample=self.sample_buttons)
        self.last_logits = logits
        return self.act_delay.push(a).numpy()
