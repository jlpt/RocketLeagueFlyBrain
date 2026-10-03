"""Head-to-head on fresh matches: imitation brain vs HyperNEAT champion (same seeds for both)."""

import json
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

from flybrain.brain.model import load_brain  # noqa: E402
from flybrain.connectome.graph import BrainGraph  # noqa: E402
from flybrain.train.hyperneat import evaluate_gains  # noqa: E402

graph = BrainGraph.load("data/brain/male_cns.npz")
base, _ = load_brain("runs/dagger/dagger_final.pt", graph)
champ, _ = load_brain("runs/hyperneat/hyperneat_final.pt", graph)
gain = champ.edge_gain.numpy() if champ.edge_gain is not None else np.ones(base.col.numel(), np.float32)
print("champion gain: min %.3f max %.3f mean %.3f" % (gain.min(), gain.max(), gain.mean()), flush=True)
n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 32
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
start = 50_000  # never used in training or selection
tot = {"base": np.zeros(3), "champion": np.zeros(3)}
per_match = {"base": [], "champion": []}
for chunk in range(0, n_seeds, 16):
    seeds = list(range(start + chunk, start + min(chunk + 16, n_seeds)))
    k = len(seeds)
    gains = np.stack([np.ones_like(gain)] * k + [gain] * k)
    res = evaluate_gains(base, gains, seeds + seeds, seconds)
    for name, sl in (("base", slice(0, k)), ("champion", slice(k, 2 * k))):
        tot[name] += [res["fitness"][sl].sum(), res["goals_for"][sl].sum(), res["goals_against"][sl].sum()]
        per_match[name] += (res["goals_for"][sl] - res["goals_against"][sl]).tolist()
    print(json.dumps({k2: v.tolist() for k2, v in tot.items()}), flush=True)
d = np.array(per_match["champion"]) - np.array(per_match["base"])
out = {
    "matches_each": n_seeds, "seconds": seconds,
    "base": {"mean_fitness": tot["base"][0] / n_seeds, "goals_for": int(tot["base"][1]), "goals_against": int(tot["base"][2])},
    "champion": {"mean_fitness": tot["champion"][0] / n_seeds, "goals_for": int(tot["champion"][1]),
                 "goals_against": int(tot["champion"][2])},
    "paired_goal_diff_gain_per_match": float(d.mean()),
    "paired_std_err": float(d.std(ddof=1) / np.sqrt(len(d))),
}
print(json.dumps(out, indent=2))
open("runs/hyperneat/confirm.json", "w").write(json.dumps(out, indent=2))
