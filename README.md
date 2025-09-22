# IsaacLab Tutorial

This tutorial provides a comprehensive guide to getting started with [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html), NVIDIA's robotics simulation framework. Learn how to set up your environment, create custom environments, and train reinforcement learning agents using this powerful simulation platform.

## Installation

See the [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for further help.

### 1. (Recommended) Pip + Source

* Create a virtual environment with `python3.11`:

    ```bash
    python3.11 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    ```

* Install `isaacsim` and its dependencies:
    ```bash
    pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
    ```

* Install `isaaclab` and its dependencies:
    ```bash
    sudo apt install cmake build-essential # -> If not installed
    git clone git@github.com:isaac-sim/IsaacLab.git
    IsaacLab/isaaclab.sh -i
    ```

### 2. Pip Only

* Create a virtual environment with `python3.11`:

    ```bash
    python3.11 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    ```

* Install `isaaclab` and `isaacsim` together:
    ```bash
    pip install isaaclab[isaacsim,all]==2.2.0 --extra-index-url https://pypi.nvidia.com
    ```

### 3. Docker


## Usage

### Training

Train a simple reinforcement learning agent (multilayer perceptron) on the Grid2d environment:

```bash
python train.py --headless --task grid2d-v0 --model mlp
```

For distributed training across multiple GPUs:

```bash
python -m torch.distributed.run \
    --rdzv-endpoint localhost:8000 \
    --nproc-per-node 2 \
    train.py \
    --headless \
    --distributed \
    --task grid2d-v0 \
    --model mlp
```

### Evaluation

Test a trained agent using a saved checkpoint:

```bash
python main.py \
    --headless \
    --task grid2d-v0 \
    --model mlp \
    --num_envs 1 \
    --checkpoint logs/grid2d/ppo/<date>/best_agent.pt
```

Replace `<date>` with the actual timestamp folder created during training.

### Command Line Options

- `--headless`: Runs the simulation without GUI rendering for faster training and evaluation.
- `--num_envs`: Number of parallel environments (default is 1024).

For complete list of available options, use:
```bash
python train.py -h    # Training options
python main.py -h     # Evaluation options
```
