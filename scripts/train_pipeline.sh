#!/usr/bin/env bash
# Resumable training pipeline: DAgger (until DAGGER_ITERS) -> ES (until ES_GENS) -> eval.
# Safe to re-run after a crash/reboot: every stage resumes from its last checkpoint.
set -u
cd "$(dirname "$0")/.."
DAGGER_ITERS=${DAGGER_ITERS:-24}
ES_GENS=${ES_GENS:-30}
mkdir -p runs/dagger runs/es

iter=$(python3 -c "
import torch, os
p='runs/dagger/dagger_latest.pt'
print(torch.load(p, weights_only=False)['extra'].get('iter', -1) if os.path.exists(p) else -1)")
if [ "$iter" -lt $((DAGGER_ITERS - 1)) ]; then
  echo "[supervise] dagger resume after iter $iter $(date)"
  ck=""
  [ "$iter" -ge 0 ] && ck="--checkpoint runs/dagger/dagger_latest.pt"
  python3 -u -m flybrain train-dagger $ck --out runs/dagger --lr 0.01 \
    --iterations "$DAGGER_ITERS" --updates-per-iter 200 >> runs/dagger/stdout2.log 2>&1 || exit 1
fi
echo "[supervise] dagger done $(date)"
cp runs/dagger/dagger_latest.pt runs/dagger/dagger_final.pt

python3 -u -m flybrain train-es --checkpoint runs/dagger/dagger_final.pt --out runs/es --resume \
  --pairs 16 --generations "$ES_GENS" --episode-seconds 45 --sigma 0.03 --lr 0.01 >> runs/es/stdout.log 2>&1 || exit 1
echo "[supervise] es done $(date)"
python3 -u -m flybrain eval --checkpoint runs/es/es_final.pt --seconds 300 --matches 4 --out runs/eval_es.json > runs/eval_es.log 2>&1
python3 -u -m flybrain eval --checkpoint runs/dagger/dagger_final.pt --seconds 300 --matches 4 --out runs/eval_dagger.json > runs/eval_dagger.log 2>&1
echo "[supervise] eval done $(date)"
