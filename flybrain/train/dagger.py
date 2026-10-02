"""Stage 1: imitation learning with DAgger, by backpropagation through the connectome.

Why this first: a 70k-neuron recurrent network controlling a car from scratch
with reward alone (pure evolution or RL) wastes almost all of its samples on
random flailing. Imitation gives the fly brain a competent prior in a few
hours of CPU time, exactly like fly-chess bootstraps from human games before
self-play. DAgger (Ross et al. 2011) fixes behaviour cloning's compounding
errors: the *fly* drives, the teacher only labels what it would have done.

Gradients flow through time (truncated BPTT) and through the fixed sparse
connectome into the per-neuron gains, biases, time constants and the sensory
and motor transductions. The wiring is never changed.

The fly sees the world through its sensory delay and acts through its motor
delay, while the teacher labels the *current* state, so the fly must learn to
anticipate -- the same thing a human player with a 200 ms reaction time does.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..brain.controller import FlyController
from ..brain.model import N_ANALOG, FlyBrain, save_brain
from ..sim.agents import BallChaseAgent, IdleAgent, TeacherAgent
from ..sim.env import Match
from ..sim.obs import OBS_DIM, build_obs
from ..sim.teacher import TeacherBot

# per-control loss weights: throttle, steer, pitch, yaw, roll | jump, boost, handbrake
ANALOG_W = torch.tensor([1.0, 2.0, 0.5, 0.5, 0.5])
BUTTON_W = torch.tensor([1.5, 1.0, 0.5])


@dataclass
class DaggerConfig:
    n_envs: int = 32
    collect_steps: int = 600            # brain steps per collection round (20 s)
    episode_seconds: float = 20.0
    iterations: int = 20
    beta0: float = 1.0                  # P(teacher drives) at iteration 0
    beta_decay: float = 0.6
    beta_min: float = 0.05
    updates_per_iter: int = 150
    batch: int = 16
    burn_in: int = 12
    window: int = 24
    lr: float = 3e-3
    reg_gain: float = 1e-4
    max_dataset_steps: int = 800_000
    opponent_mix: tuple = (0.7, 0.15, 0.15)   # teacher, ballchase, idle
    seed: int = 0


@dataclass
class Dataset:
    obs: list = field(default_factory=list)      # each (T, OBS_DIM) float16
    act: list = field(default_factory=list)      # each (T, 8) float16

    @property
    def steps(self) -> int:
        return int(sum(len(o) for o in self.obs))

    def add(self, obs: np.ndarray, act: np.ndarray) -> None:
        if len(obs) >= 8:
            self.obs.append(obs.astype(np.float16))
            self.act.append(act.astype(np.float16))

    def trim(self, max_steps: int) -> None:
        while self.steps > max_steps and len(self.obs) > 1:
            self.obs.pop(0)
            self.act.pop(0)

    def sample(self, rng: np.random.Generator, batch: int, length: int, sdelay: int, mdelay: int, burn_in: int):
        lens = np.array([len(o) for o in self.obs])
        ep = rng.choice(len(lens), size=batch, p=lens / lens.sum())
        obs = np.zeros((length, batch, OBS_DIM), np.float32)
        lab = np.zeros((length, batch, 8), np.float32)
        mask = np.zeros((length, batch), np.float32)
        for b, e in enumerate(ep):
            T = lens[e]
            s = int(rng.integers(-burn_in, T - burn_in))
            t = np.arange(s, s + length)
            oi = t - sdelay
            ok = (oi >= 0) & (oi < T)
            obs[ok, b] = self.obs[e][oi[ok]]
            li = t + mdelay
            lk = (li >= 0) & (li < T)
            lab[lk, b] = self.act[e][li[lk]]
            mask[lk, b] = 1.0
        return torch.from_numpy(obs), torch.from_numpy(lab), torch.from_numpy(mask)


def action_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    analog = torch.tanh(logits[:, :N_ANALOG])
    la = ((analog - target[:, :N_ANALOG]) ** 2 * ANALOG_W).sum(1)
    lb = (F.binary_cross_entropy_with_logits(logits[:, N_ANALOG:], target[:, N_ANALOG:], reduction="none")
          * BUTTON_W).sum(1)
    return ((la + lb) * mask).sum() / mask.sum().clamp(min=1.0)


def _make_opponent(rng, mix, seed):
    k = rng.choice(3, p=np.array(mix) / sum(mix))
    if k == 0:
        return TeacherAgent(seed=seed, aggression=float(rng.uniform(0.8, 1.1)))
    return BallChaseAgent() if k == 1 else IdleAgent()


def collect(brain: FlyBrain, cfg: DaggerConfig, beta: float, rng: np.random.Generator,
            data: Dataset, log: dict) -> None:
    """Run the fly (or, with prob beta, the teacher) in n_envs matches; label every step with the teacher."""
    M = cfg.n_envs
    matches = [Match(seed=int(rng.integers(1 << 30)), max_seconds=cfg.episode_seconds) for _ in range(M)]
    labelers = [TeacherBot(rng=np.random.default_rng(int(rng.integers(1 << 30)))) for _ in range(M)]
    opps = [_make_opponent(rng, cfg.opponent_mix, int(rng.integers(1 << 30))) for _ in range(M)]
    teacher_drives = rng.random(M) < beta
    ctrl = FlyController(brain, M)
    views = [m.reset() for m in matches]
    prev = np.zeros((M, 8), np.float32)
    ep_obs = [[] for _ in range(M)]
    ep_act = [[] for _ in range(M)]
    stats = dict(fly_goals=0, fly_conceded=0, fly_touches=0, fly_steps=0, agree=0.0)
    for step in range(cfg.collect_steps):
        obs = np.stack([build_obs(views[e][0], prev[e]) for e in range(M)])
        labels = np.stack([labelers[e].act(views[e][0], *matches[e].team_ball_prediction(0),
                                           matches[e].big_pad_flags(0)) for e in range(M)])
        fly = ctrl.step(obs)
        done_cols = []
        for e in range(M):
            applied = labels[e] if teacher_drives[e] else fly[e]
            opp_a = opps[e].act(views[e][1], matches[e])
            ep_obs[e].append(obs[e])
            ep_act[e].append(labels[e])
            views[e], info = matches[e].step([applied, opp_a])
            prev[e] = applied
            if not teacher_drives[e]:
                stats["fly_steps"] += 1
                stats["fly_touches"] += int(info["touch"][0])
                stats["agree"] += float(np.abs(fly[e][1] - labels[e][1]) < 0.3)
                if info["goal"] is not None:
                    stats["fly_goals" if info["goal"] == 0 else "fly_conceded"] += 1
            if info["done"]:
                data.add(np.array(ep_obs[e]), np.array(ep_act[e]))
                ep_obs[e], ep_act[e] = [], []
                views[e] = matches[e].reset()
                labelers[e].reset()
                opps[e].reset()
                prev[e] = 0
                done_cols.append(e)
        if done_cols:
            ctrl.reset_columns(done_cols)
    for e in range(M):
        data.add(np.array(ep_obs[e]), np.array(ep_act[e]))
    fs = max(stats["fly_steps"], 1)
    minutes = fs * matches[0].tick_skip / 120.0 / 60.0
    log.update(fly_minutes=round(minutes, 2), fly_goals=stats["fly_goals"], fly_conceded=stats["fly_conceded"],
               fly_touches_per_min=round(stats["fly_touches"] / max(minutes, 1e-6), 2),
               steer_agreement=round(stats["agree"] / fs, 3))


def train_updates(brain: FlyBrain, opt, data: Dataset, cfg: DaggerConfig, rng, n: int) -> float:
    c = brain.cfg
    losses = []
    L = cfg.burn_in + cfg.window
    for _ in range(n):
        obs, lab, mask = data.sample(rng, cfg.batch, L, c.sensory_delay, c.motor_delay, cfg.burn_in)
        with torch.no_grad():
            st = brain.init_state(cfg.batch)
            for t in range(cfg.burn_in):
                st, _ = brain.step(st, obs[t])
        st = {k: v.detach() for k, v in st.items()}
        loss = 0.0
        for t in range(cfg.burn_in, L):
            st, logits = brain.step(st, obs[t])
            loss = loss + action_loss(logits, lab[t], mask[t])
        loss = loss / cfg.window
        reg = cfg.reg_gain * (brain.theta_pre.pow(2).mean() + brain.theta_post.pow(2).mean())
        opt.zero_grad()
        (loss + reg).backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    return float(np.mean(losses))


def run_dagger(brain: FlyBrain, cfg: DaggerConfig, out_dir: str | Path, start_iter: int = 0,
               data: Dataset | None = None) -> FlyBrain:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + start_iter)
    torch.manual_seed(cfg.seed + start_iter)
    data = data or Dataset()
    opt = torch.optim.Adam(brain.param_groups(cfg.lr))
    log_path = out / "dagger_log.jsonl"
    for it in range(start_iter, cfg.iterations):
        beta = max(cfg.beta_min, cfg.beta0 * cfg.beta_decay ** it)
        t0 = time.time()
        log = {"iter": it, "beta": round(beta, 3)}
        brain.eval()
        collect(brain, cfg, beta, rng, data, log)
        data.trim(cfg.max_dataset_steps)
        t1 = time.time()
        brain.train()
        n_up = cfg.updates_per_iter
        log["loss"] = round(train_updates(brain, opt, data, cfg, rng, n_up), 4)
        log.update(dataset_steps=data.steps, collect_s=round(t1 - t0, 1), train_s=round(time.time() - t1, 1))
        print(json.dumps(log), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")
        save_brain(out / "dagger_latest.pt", brain, {"stage": "dagger", "iter": it, "dagger_cfg": asdict(cfg)})
    return brain
