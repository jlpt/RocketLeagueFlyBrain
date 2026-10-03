"""Command line entry point: ``python -m flybrain <command>`` (or ``flybrain <command>``)."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import fields
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")
warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")

DEFAULT_GRAPH = "data/brain/male_cns.npz"


def _load_graph(path):
    from .connectome.graph import BrainGraph
    if not Path(path).exists():
        raise SystemExit(f"{path} not found: run `flybrain download` and `flybrain build` first")
    return BrainGraph.load(path)


def _load_brain(args, graph):
    from .brain.model import BrainConfig, FlyBrain, load_brain
    if getattr(args, "checkpoint", None):
        brain, extra = load_brain(args.checkpoint, graph)
        return brain, extra
    return FlyBrain(graph, BrainConfig()), {}


def _add_dataclass_args(p, cls, skip=()):
    for f in fields(cls):
        if f.name in skip or f.type not in ("int", "float", "str", int, float, str):
            continue
        typ = {"int": int, "float": float, "str": str}.get(f.type, f.type)
        p.add_argument("--" + f.name.replace("_", "-"), type=typ, default=f.default)


def _dc_from_args(cls, args):
    kw = {}
    for f in fields(cls):
        if hasattr(args, f.name):
            kw[f.name] = getattr(args, f.name)
    return cls(**kw)


def cmd_download(args):
    from .connectome.download import download
    download(args.raw_dir)


def cmd_build(args):
    from .connectome.build import DEFAULT_EXCLUDE, build_graph
    g = build_graph(args.raw_dir, args.min_synapses, () if args.include_optic_lobe else DEFAULT_EXCLUDE)
    g.save(args.out)
    print(f"saved {args.out}")


def cmd_meshes(args):
    from .sim import arena_mesh
    print(arena_mesh.generate(args.out))


def cmd_import_meshes(args):
    from .sim.meshes import import_dumped_meshes
    import_dumped_meshes(args.src, args.out)


def cmd_train_dagger(args):
    import torch
    from .train.dagger import DaggerConfig, run_dagger
    torch.set_num_threads(args.threads)
    graph = _load_graph(args.graph)
    brain, extra = _load_brain(args, graph)
    cfg = _dc_from_args(DaggerConfig, args)
    start = int(extra.get("iter", -1)) + 1 if args.checkpoint and extra.get("stage") == "dagger" else 0
    run_dagger(brain, cfg, args.out, start_iter=start)


def cmd_train_es(args):
    import torch
    from .train.es import ESConfig, run_es
    torch.set_num_threads(args.threads)
    graph = _load_graph(args.graph)
    brain, extra = _load_brain(args, graph)
    cfg = _dc_from_args(ESConfig, args)
    run_es(brain, graph, cfg, args.out, resume=args.resume)


def cmd_positions(args):
    from .connectome.positions import add_positions
    add_positions(args.graph, args.raw_dir)
    print(f"added soma positions to {args.graph}")


def cmd_train_hyperneat(args):
    import torch
    from .train.hyperneat import HyperNEATConfig, run_hyperneat
    torch.set_num_threads(args.threads)
    graph = _load_graph(args.graph)
    brain, _ = _load_brain(args, graph)
    run_hyperneat(brain, graph, _dc_from_args(HyperNEATConfig, args), args.out, resume=not args.fresh)


def cmd_eval(args):
    from .eval.evaluate import evaluate
    graph = _load_graph(args.graph)
    brain, extra = _load_brain(args, graph)
    res = evaluate(brain, graph, extra, opponents=args.opponents.split(","), seconds=args.seconds,
                   n_matches=args.matches, plastic=args.plastic, seed=args.seed)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


def cmd_reaction(args):
    from .brain.reaction import measure_reaction_time
    graph = _load_graph(args.graph)
    brain, _ = _load_brain(args, graph)
    print(json.dumps(measure_reaction_time(brain, n_pairs=args.pairs), indent=2))


def main(argv=None):
    from .brain.model import BrainConfig  # noqa: F401  (import check)
    from .train.dagger import DaggerConfig

    p = argparse.ArgumentParser(prog="flybrain", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="download male CNS connectome tables (~1.1 GB)")
    s.add_argument("--raw-dir", default="data/raw")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("build", help="build the brain graph")
    s.add_argument("--raw-dir", default="data/raw")
    s.add_argument("--out", default=DEFAULT_GRAPH)
    s.add_argument("--min-synapses", type=int, default=5)
    s.add_argument("--include-optic-lobe", action="store_true")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("meshes", help="generate the approximate soccar arena meshes")
    s.add_argument("--out", default="collision_meshes")
    s.set_defaults(fn=cmd_meshes)

    s = sub.add_parser("import-meshes", help="import real meshes dumped by RLArenaCollisionDumper")
    s.add_argument("src", help="folder with the dumped .cmf files")
    s.add_argument("--out", default="collision_meshes_real")
    s.set_defaults(fn=cmd_import_meshes)

    s = sub.add_parser("train-dagger", help="stage 1: imitation learning (DAgger + BPTT)")
    s.add_argument("--graph", default=DEFAULT_GRAPH)
    s.add_argument("--checkpoint", default=None, help="resume from a checkpoint")
    s.add_argument("--out", default="runs/dagger")
    s.add_argument("--threads", type=int, default=4)
    _add_dataclass_args(s, DaggerConfig)
    s.set_defaults(fn=cmd_train_dagger)

    s = sub.add_parser("train-es", help="stage 2: evolution strategies on match reward (indirect encoding)")
    s.add_argument("--graph", default=DEFAULT_GRAPH)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", default="runs/es")
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--resume", action="store_true")
    from .train.es import ESConfig
    _add_dataclass_args(s, ESConfig)
    s.set_defaults(fn=cmd_train_es)

    s = sub.add_parser("positions", help="add 3D soma positions to the brain graph (needed for HyperNEAT)")
    s.add_argument("--graph", default=DEFAULT_GRAPH)
    s.add_argument("--raw-dir", default="data/raw")
    s.set_defaults(fn=cmd_positions)

    s = sub.add_parser("train-hyperneat", help="stage 2b: NEAT-evolved CPPN sets synapse strengths from 3D geometry")
    s.add_argument("--graph", default=DEFAULT_GRAPH)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", default="runs/hyperneat")
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--fresh", action="store_true", help="ignore existing NEAT checkpoints")
    from .train.hyperneat import HyperNEATConfig
    _add_dataclass_args(s, HyperNEATConfig)
    s.set_defaults(fn=cmd_train_hyperneat)

    s = sub.add_parser("eval", help="play matches against scripted opponents")
    s.add_argument("--graph", default=DEFAULT_GRAPH)
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--opponents", default="teacher,ballchase,idle")
    s.add_argument("--seconds", type=float, default=300.0)
    s.add_argument("--matches", type=int, default=4)
    s.add_argument("--plastic", action="store_true", help="enable dopamine plasticity during play")
    s.add_argument("--seed", type=int, default=123)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("reaction", help="measure the fly's visual reaction time")
    s.add_argument("--graph", default=DEFAULT_GRAPH)
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--pairs", type=int, default=128)
    s.set_defaults(fn=cmd_reaction)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
