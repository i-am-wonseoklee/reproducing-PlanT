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

VS Code will build the `workspace` container and start the `carla` container via Docker Compose.
This may take several minutes on the first run.

## Usage

All commands are run inside the `workspace` container (i.e., the VS Code terminal after reopening in container).

### 1. Collect training data

Drive CARLA with autopilot and save observations to disk:

```bash
python scripts/collect_data.py --host carla --port 2000 --output data/
```

### 2. Train

```bash
python train.py --config configs/plant_medium.yaml --data data/
```

### 3. Evaluate offline

```bash
python evaluate.py --config configs/plant_medium.yaml --checkpoint checkpoints/last.ckpt --data data/
```

### 4. Run closed-loop simulation

Drive CARLA with the trained model.
A BEV visualizer window opens on your host display:

```bash
python scripts/run_simulation.py --host carla --port 2000 --checkpoint checkpoints/last.ckpt
```

## Project Structure

```
.devcontainer/          Dev container config (Docker Compose + Dockerfile)
configs/                Model & training hyperparameters (MINI / SMALL / MEDIUM)
plant/
  model/                PlanT model components (one file per paper section)
  data/                 Dataset loader + route segment generation
  carla/                CARLA integration (data collector + closed-loop agent)
  utils/                Geometry utils + BEV visualizer
scripts/                Entry points for data collection and simulation
tests/                  Unit tests (runnable without CARLA)
train.py                Training entry point
evaluate.py             Offline evaluation entry point
```
