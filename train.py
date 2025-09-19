import random
import argparse

from nets import load_model, list_models
from utils import parse_args, train_cfg, get_env, load_runner

def get_options():

    #########################################################################################
    # Define argparser
    parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed used for the environment"
    )
    #########################################################################################
    # Env args
    parser.add_argument(
        "--task", type=str, default="grid2d-v0",
        help="Name of the task."
    )
    parser.add_argument(
        "--num_envs", type=int, default=None,
        help="Number of environments to simulate in parallel."
    )
    #########################################################################################
    # Model args
    parser.add_argument(
        "--model", type=str, default="mlp", choices=list_models(),
        help="Name of the model."
    )
    #########################################################################################
    # Train args
    parser.add_argument(
        "--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="The RL algorithm used for training the skrl agent."
    )
    parser.add_argument(
        "--use_runner", action="store_true", default=False,
        help="Run skrl runner, which simplifies execution but it's more constrained."
    )
    parser.add_argument(
        "--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"],
        help="The ML framework used for training the skrl agent."
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None,
        help="RL Policy training iterations."
    )
    parser.add_argument(
        "--distributed", action="store_true", default=False,
        help="Run training with multiple GPUs or nodes."
    )
    parser.add_argument(
        "--checkpoint", type=str, default='',
        help="Path to model checkpoint to resume training."
    )
    #########################################################################################
    # Video args
    parser.add_argument(
        "--video", action="store_true", default=False,
        help="Record videos during training."
    )
    parser.add_argument(
        "--video_length", type=int, default=200,
        help="Length of the recorded video (in steps)."
    )
    parser.add_argument(
        "--video_interval", type=int, default=5000,
        help="Interval between video recordings (in steps)."
    )
    #########################################################################################
    # Parse args
    opts, app_launcher = parse_args(parser, train=True)
    
    # Modify some arguments
    opts.seed = random.randint(0, 10000) if opts.seed == -1 else opts.seed
    opts.algorithm = opts.algorithm.lower()
    opts.agent_cfg_entry_point = "skrl_cfg_entry_point" if opts.algorithm in ["ppo"] else f"skrl_{opts.algorithm}_cfg_entry_point"
    return opts, app_launcher

# Get options and launch isaacsim
opts, app_launcher = get_options()

from isaaclab_tasks.utils.hydra import hydra_task_config    # Import hydra AFTER launching isaacsim
import envs.register                                        # Register new envs

@hydra_task_config(opts.task, opts.agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):
    
    # Get config data and directory for logs
    env_cfg, agent_cfg, log_dir = train_cfg(opts, env_cfg, agent_cfg, device=app_launcher.device_id) # app_launcher.local_rank

    # Create isaac env
    env = get_env(opts, env_cfg, log_dir, train=True)
    
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
        train=True
    )

    # Run training
    runner.run()

    # Close simulator
    env.close()
    print("Finished training!")

if __name__ == "__main__":
    main()
    app_launcher.app.close()
