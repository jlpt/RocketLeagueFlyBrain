# FlyBrain-RL: a Rocket League bot driven by a fruit fly brain

The policy of this bot is the wiring diagram of a real *Drosophila melanogaster*:
the **male CNS connectome** released by Janelia FlyEM and Google in 2025/26
([blog](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/),
[data](https://male-cns.janelia.org/)). The game is simulated with
[RocketSim](https://github.com/ZealanL/RocketSim). The fly is slowed down to
human reaction time, has dopamine-gated synaptic plasticity, and is trained
with imitation learning followed by evolution with an indirect encoding.

Inspired by [fly-chess](https://github.com/cesp99/fly-chess).

<!-- RESULTS -->

## How it works

```
game state ──► sensory encoding ──► 100 ms delay ──► 69,669-neuron connectome ──► DN + motor neurons ──► 33 ms delay ──► car controls
 (RocketSim)    (VPNs, mechano-       (optic lobe,        (1 synapse per 33 ms,          (readout)          (muscle)
                 sensory, internal)    not simulated)      leaky rate neurons)                ▲
                                                             ▲                             reward
                                              dopamine (PAM / PPL1 DANs) ◄──────────────────┘
```

### The brain
* **Data:** male CNS v1.0 (166k neurons, 125M synapses). We keep the proofread
  (`Traced`) neurons of the central brain + ventral nerve cord: **69,669 neurons,
  3.46M connections (>= 5 synapses), 58.7M synapses, 11,457 cell types**.
  The optic lobes are left out because we don't render images; visual
  information enters where the optic lobe hands it to the brain, at the
  visual projection neurons (VPNs).
* **Neurons:** leaky rate units, `r = tanh(relu(v))`, per-neuron time constant.
* **Synapses:** `w_ij = g_post_i * g_pre_j * sign_j * log(1 + synapses_ij) / S_i`.
  Which connections exist, their synapse counts and their signs (Dale's law,
  from predicted neurotransmitters: ACh +, GABA/Glu/histamine −) are fixed by
  the connectome. Training only changes per-neuron gains, biases and time
  constants, so the wiring of the fly stays the wiring of the fly.
* **Senses:** the observation is split by sense organ: visual features (ball,
  opponent, goals) drive 2,048 VPNs; self-motion drives 1,024
  mechanosensory/proprioceptive neurons; internal state (boost, flip
  availability, efference copy) drives 512 gustatory/chemosensory neurons.
* **Motor output:** a linear "body" reads the 1,314 descending neurons and 808
  motor neurons and turns them into throttle, steer, pitch, yaw, roll, jump,
  boost and handbrake.

### Human reaction time
A fly escapes a looming object in ~30–50 ms; people need ~200–250 ms. The
brain is time-dilated: one brain step is one synaptic hop and lasts 33 ms of
game time (4 physics ticks), about 10× slower than in the fly. Two fixed delays
are added on top: 100 ms of sensory delay (phototransduction and optic-lobe
processing, which are not simulated) and 33 ms of motor delay. The shortest
VPN → descending-neuron path is one synapse, so the hard floor is 167 ms.
`flybrain reaction` measures the real thing with a simple-reaction-time test:
hold a scene, change what the fly sees, time how long the controls take to
move halfway to their new value.

### Plasticity
Dopamine-gated three-factor learning on two sets of synapses: Kenyon cell →
mushroom body output neuron synapses (the fly's own associative-learning site,
33k synapses) and all synapses onto descending neurons (172k).

```
e_ij  <- λ e_ij + r_j (r_i − <r_i>)        eligibility trace
D      = DA(t) − <DA>                        dopamine prediction error
dW_ij <- (1 − decay) dW_ij + η D e_ij        clipped to ±|w_ij| (never flips sign)
```

`DA(t)` is the activity of the fly's own dopaminergic neurons (PAM minus
PPL1). Game reward is injected as current into those neurons, so the
dopamine signal is part of the network rather than an external number. η, λ
and the reward→dopamine gain are genes that evolution tunes. The bot can keep
learning while it plays (`--plastic`, or `FLYBRAIN_PLASTIC=1` in RLBot).

### Training: why not pure evolution?
Evolution from scratch with an indirect encoding *would* run, but a 70k-neuron
recurrent net that starts out flailing almost never produces the coherent
behaviour that evolution needs as a signal; it would take far more compute
than a CPU box has. So training has two stages, like fly-chess:

1. **Imitation (DAgger + backprop through time).** A scripted teacher bot plays;
   then the fly drives and the teacher labels what it would have done in the
   states the fly gets itself into (DAgger). Gradients flow through the
   connectome into the gains. The fly sees the world 100 ms late while the
   teacher sees it live, so the fly has to learn to anticipate.
2. **Evolution strategies with an indirect encoding.** The genome is one
   presynaptic and one postsynaptic gain per *cell type* (rare types share a
   gene with their class), plus the plasticity genes and a motor bias: about
   5,000 genes instead of 400,000 per-neuron parameters. A whole population
   runs as one sparse matrix product, fitness is real match outcome against
   the teacher, and the algorithm is OpenAI-ES (antithetic pairs, common
   random numbers, centred ranks, Adam).

### The arena
RocketSim needs Rocket League's arena collision meshes, which have to be
dumped from the game. Without them we generate an approximation from the
published dimensions: goals, 45° corners, curved floor/wall/ceiling
transitions. Cars drive up walls, the ball bounces, goals score. If you own
the game, dump the real meshes and use them (see below).

## Quick start

```bash
pip install -e .[dev]
python -m flybrain download            # 1.1 GB of connectome tables (CC-BY 4.0)
python -m flybrain build               # -> data/brain/male_cns.npz
python -m flybrain train-dagger --out runs/dagger
python -m flybrain train-es --checkpoint runs/dagger/dagger_latest.pt --out runs/es
python -m flybrain eval --checkpoint runs/es/es_final.pt
python -m flybrain reaction --checkpoint runs/es/es_final.pt
python -m flybrain.eval.replay --checkpoint runs/es/es_final.pt   # 3D replay viewer (HTML)
pytest
```

The trained brain is in `checkpoints/flybrain.pt` and the compact connectome
graph is in `checkpoints/male_cns_graph.npz`, so evaluation and play work
without the download:

```bash
python -m flybrain eval --graph checkpoints/male_cns_graph.npz --checkpoint checkpoints/flybrain.pt
```

### Using the real arena
1. On Windows, download [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper/releases).
2. Start Rocket League, go into Free Play, run the dumper.
3. `python -m flybrain import-meshes path/to/collision-meshes` (verifies the 16 standard soccar meshes by hash)
4. `export RS_COLLISION_MESHES=collision_meshes_real`

### Playing it in Rocket League
`rlbot/fly_bot.cfg` is an [RLBot](https://rlbot.org) (v4, `pip install rlbot`)
bot. Load it in RLBotGUI and start a match. It runs the same 30 Hz brain with
the same delays. Set `FLYBRAIN_PLASTIC=1` to let it keep learning across games.
This path has not been tested against the real game from this repo's CI.

## Layout
```
flybrain/connectome   download + build the brain graph (male CNS), synthetic test graph
flybrain/brain        FlyBrain model, delays, plasticity, reaction-time test, batched controller
flybrain/sim          RocketSim match, generated arena meshes, sensory encoding, teacher + baseline bots
flybrain/train        DAgger (stage 1), evolution strategies (stage 2), batched rollouts
flybrain/eval         match evaluation, replay recorder + 3D viewer
flybrain/play         single-fly agent, RLBot packet conversion
rlbot/                RLBot bot config
```

## Licences
Code: MIT. Connectome data: FlyEM male CNS, CC-BY 4.0. Please cite the male
CNS paper (Cell, 2026; doi:10.1016/j.cell.2026.08.015) if you use the brain.
