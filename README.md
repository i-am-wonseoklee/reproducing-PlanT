# reproducing-PlanT

A PyTorch reproduction of the PlanT paper.

> Renz et al., "PlanT: Explainable Planning Transformers via Object-level Representations," arXiv:2210.14222, 2022

The entire development environment (including the CARLA server) runs inside Docker containers.
The host only needs Docker and the NVIDIA Container Toolkit.

## Host Prerequisites

| Requirement | Notes |
|---|---|
| [Docker Engine](https://docs.docker.com/engine/install/) | |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | GPU pass-through for both PyTorch and CARLA |
| [VS Code](https://code.visualstudio.com/) + [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) | |
| X11 display server | Already present on Linux desktops |

## Quickstart

**1. Allow containers to access your X display** (run once per session):

```bash
xhost +local:docker
```

**2. Open the repo in VS Code and reopen in container:**

```
Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

VS Code will build the `workspace` container.
This may take several minutes on the first run.

The `carla` container does **not** start automatically.
Start and stop it on demand from the VS Code terminal or the Tasks menu:

```
Ctrl+Shift+P → "Tasks: Run Task" → "Start CARLA" / "Stop CARLA"
```

## Usage

All commands are run inside the `workspace` container (i.e., the VS Code terminal after reopening in container).

### 1. Collect training data

Drive CARLA with autopilot and save observations to disk:

```bash
python3 scripts/collect.py \
    --config configs/collector.yaml \  # CARLA connection and spawn settings
    --output data/frames.db \          # output SQLite DB path
    --episodes 10 \                    # number of episodes to collect
    --ticks 2000                       # ticks per episode
```

### 2. Train

```bash
python3 scripts/train.py \
    --model-config configs/plant.yaml \   # model architecture (d_model, n_layers, …)
    --train-config configs/train.yaml \   # training loop (lr, batch_size, epochs, …)
    --data data/frames.db \               # path to collected frames DB
    --run-name my_run                     # (optional) subdirectory under log/ckpt dirs
```

### 3. Evaluate offline

```bash
python3 scripts/evaluate.py \
    --checkpoint checkpoints/best.ckpt \  # trained checkpoint (model config is embedded)
    --data data/frames.db \               # path to frames DB
    --out data/eval.db \                  # (optional) output DB for metrics and BEV images
    --episode episode_0001                # (optional) render a single episode only
```

### 4. Run closed-loop simulation

Drive CARLA with the trained model.
A BEV visualizer window opens on your host display:

```bash
# not yet implemented
# python3 scripts/simulate.py --checkpoint checkpoints/best.ckpt
```