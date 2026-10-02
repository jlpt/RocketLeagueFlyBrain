"""Sensory encoding: game state -> three sensory channels of the fly.

The observation is split by the kind of sense organ it would come from, and
each part drives a different population of real sensory neurons:

  * visual   -> visual projection neurons (VPNs; optic-lobe output to the brain):
                where the ball / opponent / goals are and how they move,
                egocentric (car frame) plus allocentric (field frame).
  * self     -> mechanosensory / proprioceptive neurons (the fly's
                "vestibular" system: halteres, chordotonal organs):
                own velocity, rotation, orientation, ground contact.
  * internal -> gustatory / chemosensory neurons (internal state):
                boost "fuel", jump/flip availability, efference copy of the
                last motor command.
"""

from __future__ import annotations

import numpy as np

from .state import (ARENA_X, ARENA_Y, ARENA_Z, CAR_MAX_ANGVEL, CAR_MAX_SPEED,
                    OPP_GOAL, OWN_GOAL, View)

VISUAL_DIM = 28
SELF_DIM = 17
INTERNAL_DIM = 13
OBS_DIM = VISUAL_DIM + SELF_DIM + INTERNAL_DIM
GROUP_SLICES = {
    "in_visual": slice(0, VISUAL_DIM),
    "in_self": slice(VISUAL_DIM, VISUAL_DIM + SELF_DIM),
    "in_internal": slice(VISUAL_DIM + SELF_DIM, OBS_DIM),
}
_POS_NORM = np.array([ARENA_X, ARENA_Y, ARENA_Z])


def build_obs(view: View, prev_action: np.ndarray) -> np.ndarray:
    me, b = view.me, view.ball_pos
    rot = me.rot
    rel_b = b - me.pos
    dist_b = np.linalg.norm(rel_b) + 1e-6
    rel_b_loc = rot @ rel_b
    v = np.empty(OBS_DIM, dtype=np.float32)
    # ---------------- visual (28)
    v[0:3] = rel_b_loc / CAR_MAX_SPEED
    v[3:6] = rot @ (view.ball_vel - me.vel) / CAR_MAX_SPEED
    v[6:9] = rel_b_loc / dist_b
    v[9] = min(dist_b / 5000.0, 2.0)
    v[10:13] = rot @ (OPP_GOAL - me.pos) / 5000.0
    v[13:16] = rot @ (OWN_GOAL - me.pos) / 5000.0
    if view.opp is not None and not view.opp.demoed:
        v[16:19] = rot @ (view.opp.pos - me.pos) / CAR_MAX_SPEED
        v[19:22] = rot @ (view.opp.vel - me.vel) / CAR_MAX_SPEED
    else:
        v[16:22] = 0.0
    v[22:25] = b / _POS_NORM
    v[25:28] = view.ball_vel / CAR_MAX_SPEED
    # ---------------- self motion (17)
    s = VISUAL_DIM
    v[s:s + 3] = me.pos / _POS_NORM
    v[s + 3:s + 6] = rot @ me.vel / CAR_MAX_SPEED
    v[s + 6:s + 9] = rot @ me.ang_vel / CAR_MAX_ANGVEL
    v[s + 9:s + 12] = me.fwd
    v[s + 12:s + 15] = me.up
    v[s + 15] = float(me.on_ground)
    v[s + 16] = np.linalg.norm(me.vel) / CAR_MAX_SPEED
    # ---------------- internal (13)
    s = VISUAL_DIM + SELF_DIM
    v[s] = me.boost / 100.0
    v[s + 1] = float(me.has_flip)
    v[s + 2] = float(me.has_jumped)
    v[s + 3] = float(me.is_flipping)
    v[s + 4:s + 12] = prev_action
    v[s + 12] = 1.0
    return v
