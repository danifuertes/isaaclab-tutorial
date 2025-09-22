import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skrl.utils.spaces.torch import unflatten_tensorized_space

from nets import load_model, list_models
from utils import parse_args, play_cfg, get_env, load_runner, take_action


def get_options():

    #########################################################################################
    # Define argparser
    parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed used for the environment"
    )
    #########################################################################################
    # Env args
    parser.add_argument(
        "--task", type=str, default="grid2d-v0", help="Name of the task."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of environments to simulate in parallel.",
    )
    #########################################################################################
    # Model args
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=list_models(),
        help="Name of the model.",
    )
    #########################################################################################
    # Train args
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="The RL algorithm used for training the skrl agent.",
    )
    parser.add_argument(
        "--use_runner",
        action="store_true",
        default=False,
        help="Run skrl runner, which simplifies execution but it's more constrained.",
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax", "jax-numpy"],
        help="The ML framework used for training the skrl agent.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="RL Policy training iterations.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Run training with multiple GPUs or nodes.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training.",
    )
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        help="Use the pre-trained checkpoint from Nucleus.",
    )
    #########################################################################################
    # Video args
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record videos."
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=1000,
        help="Length of the recorded video (in steps).",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=5000,
        help="Interval between video recordings (in steps).",
    )
    #########################################################################################
    # Execution args
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Run in real-time, if possible.",
    )
    #########################################################################################
    # Parse args
    opts, app_launcher = parse_args(parser)

    # Modify some arguments
    opts.seed = random.randint(0, 10000) if opts.seed == -1 else opts.seed
    opts.algorithm = opts.algorithm.lower()
    opts.agent_cfg_entry_point = (
        "skrl_cfg_entry_point"
        if opts.algorithm in ["ppo"]
        else f"skrl_{opts.algorithm}_cfg_entry_point"
    )
    return opts, app_launcher

# Get options and launch isaacsim
opts, app_launcher = get_options()
import envs.register  # Register new envs
from isaaclab.sim import SimulationCfg, SimulationContext

def main():
    """Play with skrl agent."""

    # Config play
    env_cfg, agent_cfg, log_dir = play_cfg(opts)

    # Create isaac env
    env = get_env(opts, env_cfg, log_dir)

    # Load model
    model = load_model(opts.model, env)

    # Get skrl runner
    runner = load_runner(
        env,
        agent_cfg,
        model=model,
        use_runner=opts.use_runner,
        ml_framework=opts.ml_framework,
        checkpoint=opts.checkpoint,
    )

    # Reset environment
    obs, _ = env.reset()
    obs_unflatten = unflatten_tensorized_space(env.observation_space, obs)
    path = [obs_unflatten['agent'].cpu().numpy() * (env.cfg.grid_size - 1)]
    goal = obs_unflatten['goal'].cpu().numpy() * (env.cfg.grid_size - 1)
    
    # Simulate environments
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    while app_launcher.app.is_running() and not done.all():

        # Take action
        actions = take_action(env, runner, obs)

        # Environment update
        obs, _, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        
        # Collect results
        obs_unflatten = unflatten_tensorized_space(env.observation_space, obs)
        path.append(obs_unflatten['agent'].cpu().numpy() * (env.cfg.grid_size - 1))

    # Plot results
    path.pop(-1)
    path.append(goal)
    path = np.stack(path, axis=1)
    plot_path(path, grid_size=env.cfg.grid_size, index=0)

    # Close the simulator
    env.close()
    print("Finished!")

def plot_path(path, grid_size: int = 5, index: int = 0):
    """
    Plot the trajectory of a single agent in the grid.
    
    Args:
        results (np.ndarray): Contains a (num_envs, num_steps, 2) array of agent positions.
        grid_size (int): Size of the grid. Defaults to 5.
        index (int): Which environment in the batch to plot. Defaults to 0.
    """
    path = path[index]

    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    
    # Plot agent trajectory
    plt.plot(path[:, 0], path[:, 1], marker='o', color='blue', label="Path")
    
    # Start and goal markers
    plt.scatter(path[0, 0], path[0, 1], color='green', s=100, marker='s', label="Start", zorder=5)
    plt.scatter(path[-1, 0], path[-1, 1], color='orange', s=140, marker='*', label="Goal", zorder=5)
    
    plt.legend()
    plt.title(f"Agent {index + 1} trajectory")
    plt.show()

if __name__ == "__main__":
    main()
    app_launcher.app.close()
