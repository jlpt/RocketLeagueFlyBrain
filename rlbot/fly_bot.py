"""RLBot (v4 framework, ``pip install rlbot``) bot driven by the fly brain.

Add ``rlbot/fly_bot.cfg`` in RLBotGUI (Add -> Load folder / bot cfg) and start
a match. The fly runs at 30 brain steps per second (one synaptic hop each)
with the same 100 ms sensory + 33 ms motor delays used in training, so its
reaction time in the real game is the same ~200-250 ms.

Set FLYBRAIN_PLASTIC=1 to let it keep learning (dopamine plasticity) while it
plays; the learned weight changes are written to ``plastic_state.pt`` on
shutdown and reloaded next time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState  # noqa: E402
from rlbot.utils.structures.game_data_struct import GameTickPacket  # noqa: E402

from flybrain.play.fly_agent import FlyAgent  # noqa: E402
from flybrain.play.packet import view_from_packet  # noqa: E402
from flybrain.sim.env import step_reward  # noqa: E402

GRAPH = os.environ.get("FLYBRAIN_GRAPH", str(ROOT / "checkpoints" / "male_cns_graph.npz"))
CHECKPOINT = os.environ.get("FLYBRAIN_CHECKPOINT", str(ROOT / "checkpoints" / "flybrain.pt"))
PLASTIC = os.environ.get("FLYBRAIN_PLASTIC", "0") == "1"
STEP_SECONDS = 4 / 120.0


class FlyBrainBot(BaseAgent):
    def initialize_agent(self):
        import torch
        torch.set_num_threads(2)
        self.fly = FlyAgent.from_files(GRAPH, CHECKPOINT, plastic=PLASTIC)
        state = ROOT / "plastic_state.pt"
        if PLASTIC and state.exists() and self.fly.ctrl.plasticity is not None:
            self.fly.ctrl.plasticity.dW.copy_(torch.load(state)["dW"])
        self.trackers = {}
        self.last_t = -1.0
        self.controls = SimpleControllerState()
        self.prev_view = None
        self.last_score = None
        self.last_touch = 0.0
        self._in_kickoff = False

    def _reward(self, packet: GameTickPacket, view) -> float | None:
        if not PLASTIC or self.prev_view is None:
            return None
        score = (packet.teams[0].score, packet.teams[1].score)
        goal = None
        if self.last_score is not None and score != self.last_score:
            goal = 0 if score[0] > self.last_score[0] else 1     # team that scored
        self.last_score = score
        touch = packet.game_ball.latest_touch
        touched = touch.player_index == self.index and touch.time_seconds > self.last_touch
        if touched:
            self.last_touch = touch.time_seconds
        return step_reward(view, self.prev_view, touched, goal)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        t = packet.game_info.seconds_elapsed
        if t - self.last_t < STEP_SECONDS - 1e-4 and self.last_t >= 0:
            return self.controls
        self.last_t = t
        view = view_from_packet(packet, self.index, self.trackers)
        if packet.game_info.is_kickoff_pause and self.prev_view is not None and not self._in_kickoff:
            self.fly.reset()
        self._in_kickoff = bool(packet.game_info.is_kickoff_pause)
        a = self.fly.act(view, reward=self._reward(packet, view))
        self.prev_view = view
        c = SimpleControllerState()
        c.throttle, c.steer, c.pitch, c.yaw, c.roll = (float(np.clip(x, -1, 1)) for x in a[:5])
        c.jump, c.boost, c.handbrake = bool(a[5] > 0.5), bool(a[6] > 0.5), bool(a[7] > 0.5)
        self.controls = c
        return c

    def retire(self):
        if PLASTIC and self.fly.ctrl.plasticity is not None:
            import torch
            torch.save({"dW": self.fly.ctrl.plasticity.dW}, ROOT / "plastic_state.pt")
