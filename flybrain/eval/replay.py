"""Record matches (including the fly's descending-neuron activity) and build the 3D replay viewer.

    python -m flybrain.eval.replay --checkpoint checkpoints/flybrain.pt --out runs/replays/index.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..sim.agents import Agent, BallChaseAgent, IdleAgent, TeacherAgent
from ..sim.env import Match, to_controls

TEMPLATE = Path(__file__).with_name("viewer_template.html")
N_RASTER = 48


def _flat(a, scale=1.0):
    return np.round(np.asarray(a) * scale).astype(int).reshape(-1).tolist()


def record(blue: Agent, orange: Agent, seconds: float, seed: int, title: str,
           blue_name: str, orange_name: str, post_goal_steps: int = 36) -> dict:
    m = Match(seed=seed, max_seconds=1e9)
    agents = [blue, orange]
    views = m.reset("kickoff")
    for a in agents:
        a.reset()
    ball, cars = [], [dict(pos=[], fwd=[], up=[], boost=[], ctrl=[]) for _ in range(2)]
    dn_frames = []
    fly = blue if hasattr(blue, "ctrl") else None
    fly_dn_idx = getattr(fly, "raster_idx", None) if fly is not None else None
    if fly is not None and fly_dn_idx is None:
        fly_dn_idx = fly.ctrl.brain.idx_out
    goals = []
    score = [0, 0]

    def snap(acts):
        gs = m.arena.get_gym_state()
        ball.append(gs[2][0][0:3])
        for i in range(2):
            c = gs[3 + i][0]
            cars[i]["pos"].append(c[11:14])
            cars[i]["fwd"].append(c[24:27])
            cars[i]["up"].append(c[30:33])
            cars[i]["boost"].append(c[10])
            a = acts[i]
            cars[i]["ctrl"].append([a[0], a[1], a[5], a[6]])
        if fly is not None:
            dn_frames.append(fly.ctrl.state["r"][fly_dn_idx, 0].numpy().copy())

    n = int(seconds * 120 / m.tick_skip)
    step = 0
    while step < n:
        acts = [ag.act(v, m) for ag, v in zip(agents, views)]
        views, info = m.step(acts)
        snap(acts)
        step += 1
        if info["goal"] is not None:
            score[info["goal"]] += 1
            goals.append({"frame": len(ball) - 1, "team": int(info["goal"]), "score": list(score)})
            zero = [np.zeros(8), np.zeros(8)]
            for c in m.cars:
                c.set_controls(to_controls(np.zeros(8)))
            for _ in range(post_goal_steps):
                m.arena.step(m.tick_skip)
                snap(zero)
            views = m.reset("kickoff")
            for a in agents:
                a.reset()

    rep = {
        "title": title, "fps": 30, "frames": len(ball), "blue": blue_name, "orange": orange_name,
        "ball": _flat(ball), "goals": goals, "score": score,
        "cars": [{"team": i, "pos": _flat(c["pos"]), "fwd": _flat(c["fwd"], 100), "up": _flat(c["up"], 100),
                  "boost": _flat(c["boost"]), "ctrl": _flat(c["ctrl"], 100)} for i, c in enumerate(cars)],
    }
    if fly is not None and dn_frames:
        D = np.stack(dn_frames)                      # (T, n_dn)
        var = D.var(0)
        top = np.argsort(-var)[:N_RASTER]
        top = top[np.argsort(-D[:, top].mean(0))]
        q = np.clip((D[:, top] * 10).astype(int), 0, 9)
        rep["raster"] = ["".join(map(str, row)) for row in q]
        names = getattr(fly, "neuron_names", None)
        rep["raster_names"] = [str(names[int(fly_dn_idx[i])]) for i in top] if names is not None else []
    return rep


DN_NOTES = {
    "DNp01": "giant fiber (escape jump)",
    "DNa01": "steering",
    "DNa02": "steering",
    "MDN": "moonwalker (backing up)",
    "DNp09": "forward walking",
    "DNg02": "wing power",
}


def arena_meta() -> dict:
    m = Match(seed=0)
    pads = [[round(p.get_pos().x), round(p.get_pos().y), int(p.is_big)] for p in m.arena.get_boost_pads()]
    return {"pads": pads, "dn_notes": DN_NOTES}


def build_viewer(replays: list[dict], out: str | Path, meta: dict | None = None) -> Path:
    meta = {**arena_meta(), **(meta or {})}
    html = TEMPLATE.read_text()
    payload = json.dumps({"replays": replays, "meta": meta or {}}, separators=(",", ":"))
    html = html.replace("/*__REPLAY_DATA__*/null", payload)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


OPPONENTS = {"teacher": ("Teacher bot", lambda s: TeacherAgent(seed=s)),
             "ballchase": ("Ball-chaser", lambda s: BallChaseAgent()),
             "idle": ("Idle car", lambda s: IdleAgent())}


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="data/brain/male_cns.npz")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--label", default="Fly brain")
    p.add_argument("--opponents", default="teacher,ballchase")
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--teacher-match", action="store_true", help="also record teacher vs ball-chaser")
    p.add_argument("--json-out", default=None, help="also save the raw replays as json")
    p.add_argument("--out", default="runs/replays/index.html")
    p.add_argument("--note", default="Simulated in RocketSim on a generated approximation of the soccar arena. "
                   "The fly sees the game 100 ms late and acts through a 33 ms motor delay, one synapse per 33 ms.")
    args = p.parse_args(argv)
    replays = []
    if args.checkpoint:
        from ..play.fly_agent import FlyAgent
        for k, opp in enumerate(args.opponents.split(",")):
            fly = FlyAgent.from_files(args.graph, args.checkpoint)
            import torch
            from ..connectome.graph import BrainGraph
            g = BrainGraph.load(args.graph)
            fly.neuron_names = np.where(g.cell_type != "", g.cell_type, g.superclass)
            fly.raster_idx = torch.from_numpy(g.groups["out_dn"].astype(np.int64))
            name, mk = OPPONENTS[opp]
            print(f"recording fly vs {opp} ...", flush=True)
            replays.append(record(fly, mk(args.seed + k), args.seconds, args.seed + k,
                                  f"{args.label} vs {name}", args.label, name))
    if args.teacher_match or not args.checkpoint:
        print("recording teacher vs ball-chaser ...", flush=True)
        replays.append(record(TeacherAgent(seed=1), BallChaseAgent(), args.seconds, args.seed + 99,
                              "Teacher bot vs Ball-chaser", "Teacher bot", "Ball-chaser"))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(replays))
    print(build_viewer(replays, args.out, {"note": args.note}))


if __name__ == "__main__":
    main()
