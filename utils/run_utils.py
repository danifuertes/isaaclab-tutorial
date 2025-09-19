import os
import time
import copy
import skrl
import torch
import shutil
import argparse
import gymnasium as gym
from pathlib import Path
from datetime import datetime

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.trainers.torch import Trainer
from skrl.utils.runner.torch import Runner

def load_runner(
    env: gym.Env,
    agent_cfg: dict,
    model: torch.nn.Module = None,
    use_runner: bool = False,
    ml_framework: str = 'torch',
    checkpoint: str = '',
    train: bool = False
) -> Trainer | Runner:
    """
    Load and configure a SKRL runner/trainer for a given environment and agent configuration.

    Args:
        env (gym.Env): The environment instance.
        agent_cfg (dict): Agent configuration dictionary.
        model (torch.nn.Module, optional): Custom model to use. Defaults to None.
        use_runner (bool, optional): Whether to use a runner or trainer. Defaults to False.
        ml_framework (str, optional): Machine learning framework ('torch' supported). Defaults to 'torch'.
        checkpoint (str, optional): Path to checkpoint file to load. Defaults to ''.
        train (bool, optional): Whether to set the runner in training mode. Defaults to False.

    Returns:
        skrl.trainers.torch.Trainer or skrl.utils.runner.torch.Runner: Configured runner or trainer.
    """
    assert ml_framework == 'torch', f"Currently, only torch framework is supported, got {ml_framework}"
    
    # Define runner
    if use_runner:
        runner = get_runner(env, agent_cfg, model=model, ml_framework=ml_framework, train=train)
    else:
        runner = get_trainer(model, env, agent_cfg)
    
    # If not training, set eval mode
    if not train:
        runner.agent.set_running_mode("eval")

    # Load checkpoint
    # from isaaclab.utils.assets import retrieve_file_path
    # resume_path = retrieve_file_path(checkpoint) if checkpoint else None
    if os.path.isfile(checkpoint):
        runner.agent.load(checkpoint)
        print(f"[INFO] Model checkpoint loaded from: {checkpoint}")
    
    return runner

def get_runner(
    env: gym.Env,
    cfg: dict,
    model: torch.nn.Module = None,
    ml_framework: str = 'torch',
    train: bool = False
) -> Runner:
    """
    Create and configure a SKRL runner for the specified environment and configuration.

    Args:
        env (gym.Env): The environment instance.
        cfg (dict): Agent configuration dictionary.
        model (torch.nn.Module, optional): Custom model to use. Defaults to None.
        ml_framework (str, optional): Machine learning framework ('torch' supported). Defaults to 'torch'.
        train (bool, optional): Whether to set the runner in training mode. Defaults to False.

    Returns:
        skrl.utils.runner.torch.Runner: Configured runner instance.
    """
    
    # Ensure YAML has enough info to create a trainer
    assert 'memory' in cfg, "Memory is missing in config. Define a memory (https://skrl.readthedocs.io/en/latest/api/memories.html) \
        in the corresponding YAML located in envs/agents"
    assert 'agent' in cfg, "Agent is missing in config. Define an agent (https://skrl.readthedocs.io/en/latest/api/agents.html) \
        in the corresponding YAML located in envs/agents"
    assert 'trainer' in cfg, "Trainer is missing in config. Define a trainer (https://skrl.readthedocs.io/en/latest/api/trainers.html) \
        in the corresponding YAML located in envs/agents"
    assert 'models' in cfg, "Models are missing in config. Define models (https://skrl.readthedocs.io/en/latest/api/models.html) \
        in the corresponding YAML located in envs/agents"
    
    # Get and configure skrl runner: https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    Runner = _get_skrl_runner(ml_framework, train=train)
    runner = Runner(env, cfg)
    
    # Load default model
    if model is None:
        print("[INFO] No model provided, using default model in YAML.")
    
    # Load custom model
    else:
        runner._models = {'agent': model}
        runner._agent = runner._generate_agent(runner._env, copy.deepcopy(runner._cfg), runner._models)
        runner._trainer = runner._generate_trainer(runner._env, copy.deepcopy(runner._cfg), runner._agent)
    return runner
    
def _get_skrl_runner(ml_framework: str = 'torch', train: bool = False) -> Runner:
    """
    Get the SKRL runner class for the specified machine learning framework.

    Args:
        ml_framework (str, optional): Machine learning framework ('torch' or 'jax'). Defaults to 'torch'.
        train (bool, optional): Whether to set the runner in training mode. Defaults to False.

    Returns:
        skrl.utils.runner.torch.Runner: Runner class for the specified framework.
    """
    
    # Get skrl runner (torch or jax)
    if ml_framework.startswith("torch"):
        from skrl.utils.runner.torch import Runner
    elif ml_framework.startswith("jax"):
        from skrl.utils.runner.jax import Runner
    
    # Configure the ML framework into the global skrl variable
    if not train:
        if ml_framework.startswith("jax"):
            skrl.config.jax.backend = "jax" if ml_framework == "jax" else "numpy"
    return Runner

def get_trainer(model: torch.nn.Module, env: gym.Env, cfg: dict) -> Trainer:
    """
    Create and configure a SKRL trainer for the specified model, environment, and configuration.
    See https://skrl.readthedocs.io/en/latest/api/trainers.html for further help.

    Args:
        model (torch.nn.Module): Model to use for the agent.
        env (gym.Env): The environment instance.
        cfg (dict): Configuration dictionary for the agent.

    Returns:
        skrl.trainers.torch.Trainer: Configured trainer instance.
    """
    
    # Ensure YAML has enough info to create a trainer
    assert 'memory' in cfg, "Memory is missing in config. Define a memory (https://skrl.readthedocs.io/en/latest/api/memories.html) \
        in the corresponding YAML located in envs/agents"
    assert 'agent' in cfg, "Agent is missing in config. Define an agent (https://skrl.readthedocs.io/en/latest/api/agents.html) \
        in the corresponding YAML located in envs/agents"
    assert 'trainer' in cfg, "Trainer is missing in config. Define a trainer (https://skrl.readthedocs.io/en/latest/api/trainers.html) \
        in the corresponding YAML located in envs/agents"
    
    # Separate cfgs
    memory_cfg = cfg['memory']
    agent_cfg = cfg['agent']
    trainer_cfg = cfg['trainer']
    
    # Get classes
    memory_class = _get_memory(memory_cfg['class'])
    agent_class, default_cfg = _get_agent(agent_cfg['class'])
    trainer_class = _get_trainer(trainer_cfg['class'])
    
    # Remove class key
    del memory_cfg['class']
    del agent_cfg['class']
    del trainer_cfg['class']
    
    # Parse memory args
    memory_size = memory_cfg.get('memory_size', None)
    if memory_size is None or memory_size < 0:
        memory_cfg['memory_size'] = agent_cfg['rollouts'] if 'rollouts' in agent_cfg else default_cfg['rollouts']
    
    # Define memory
    memory = memory_class(
        memory_size=memory_cfg['memory_size'],
        num_envs=env.num_envs,
        device=env.device
    )
    
    # Parse agent args
    for k, v in agent_cfg.items():
        assert k in default_cfg, f"Expected all keys for agent {agent_class} to be in {default_cfg.keys()}, got {k}"
        if k == 'learning_rate_scheduler' and v is not None:
            v = _get_lr_scheduler(v)
        default_cfg[k] = v
    
    # Define agent
    agent = agent_class(
        models=model,
        memory=memory,
        cfg=default_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device
    )
    
    # Parse trainer args
    trainer_cfg['headless'] = cfg['headless']
    
    # Define trainer
    trainer = trainer_class(
        env=env,
        agents=agent,
        cfg=trainer_cfg
    )
    trainer.run = trainer.train
    trainer.agent = trainer.agents
    return trainer

def _get_memory(memory: str) -> Memory:
    """
    Get the SKRL memory class based on the provided memory name.
    See https://skrl.readthedocs.io/en/latest/api/memories.html for further help.

    Args:
        memory (str): Name of the memory class.

    Returns:
        skrl.memories.torch.Memory: Memory class.
    """
    if memory == 'RandomMemory':
        from skrl.memories.torch import RandomMemory
        return RandomMemory
    else:
        assert False, f"Expected memory to be in be an available memory from skrl (https://skrl.readthedocs.io/en/latest/api/memories.html), got {memory}"

def _get_agent(agent: str) -> Agent:
    """
    Get the SKRL agent class and its default configuration based on the provided agent name.
    See https://skrl.readthedocs.io/en/latest/api/agents.html for further help.

    Args:
        agent (str): Name of the agent class.

    Returns:
        tuple: (Agent class, default configuration dictionary)
    """
    if agent == 'A2C':
        from skrl.agents.torch.a2c import A2C, A2C_DEFAULT_CONFIG
        return A2C, A2C_DEFAULT_CONFIG.copy()
    elif agent == 'AMP':
        from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
        return AMP, AMP_DEFAULT_CONFIG.copy()
    elif agent == 'CEM':
        from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG
        return CEM, CEM_DEFAULT_CONFIG.copy()
    elif agent == 'DDPG':
        from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
        return DDPG, DDPG_DEFAULT_CONFIG.copy()
    elif agent == 'DDQN':
        from skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG
        return DDQN, DDQN_DEFAULT_CONFIG.copy()
    elif agent == 'DQN':
        from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
        return DQN, DQN_DEFAULT_CONFIG.copy()
    elif agent == 'PPO':
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        return PPO, PPO_DEFAULT_CONFIG.copy()
    elif agent == 'Q_LEARNING':
        from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG
        return Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG.copy()
    elif agent == 'RPO':
        from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG
        return RPO, RPO_DEFAULT_CONFIG.copy()
    elif agent == 'SAC':
        from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
        return SAC, SAC_DEFAULT_CONFIG.copy()
    elif agent == 'SARSA':
        from skrl.agents.torch.sarsa import SARSA, SARSA_DEFAULT_CONFIG
        return SARSA, SARSA_DEFAULT_CONFIG.copy()
    elif agent == 'TD3':
        from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
        return TD3, TD3_DEFAULT_CONFIG.copy()
    elif agent == 'TRPO':
        from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG
        return TRPO, TRPO_DEFAULT_CONFIG.copy()
    else:
        assert False, f"Expected agent to be an available agent from skrl (https://skrl.readthedocs.io/en/latest/api/agents.html), got {agent}"

def _get_lr_scheduler(scheduler: str) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get the learning rate scheduler class based on the provided scheduler name.
    See https://skrl.readthedocs.io/en/latest/api/resources/schedulers.html for further help.

    Args:
        scheduler (str): Name of the scheduler.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: Scheduler class.
    """
    if scheduler == 'KLAdaptiveLR':
        from skrl.resources.schedulers.torch import KLAdaptiveLR
        return KLAdaptiveLR
    elif scheduler == 'StepLR':
        from torch.optim.lr_scheduler import StepLR
        return StepLR
    else:
        assert False, f"Currently, just these schedulers are supported: [KLAdaptiveLR, StepLR], got {scheduler}"

def _get_trainer(trainer: str) -> Trainer:
    """
    Get the SKRL trainer class based on the provided trainer name.
    See https://skrl.readthedocs.io/en/latest/api/trainers.html for further help.

    Args:
        trainer (str): Name of the trainer class.

    Returns:
        skrl.trainers.torch.Trainer: Trainer class.
    """
    if trainer == 'SequentialTrainer':
        from skrl.trainers.torch import SequentialTrainer
        return SequentialTrainer
    elif trainer == 'ParallelTrainer':
        from skrl.trainers.torch import ParallelTrainer
        return ParallelTrainer
    elif trainer == 'StepTrainer':
        from skrl.trainers.torch import StepTrainer
        return StepTrainer
    else:
        assert False, f"Expected trainer to be an available trainer from skrl (https://skrl.readthedocs.io/en/latest/api/trainers.html), got {trainer}"
    
def train_cfg(opts: argparse.ArgumentParser, env_cfg: dict, agent_cfg: dict, device: torch.device) -> tuple[dict, dict, str]:
    """
    Configure and prepare the environment and agent for training, including logging and seeding.

    Args:
        opts (argparse.ArgumentParser): Command-line options.
        env_cfg (dict): Environment configuration.
        agent_cfg (dict): Agent configuration.
        device (torch.device): Device to use for training.

    Returns:
        tuple: (Updated environment config, updated agent config, log directory path)
    """
    from skrl.utils import set_seed
    from isaaclab.utils.io import dump_pickle, dump_yaml

    # Set agent and env seed. Note: certain randomization occur in the env initialization, so we set the seed here
    agent_cfg["seed"] = opts.seed if opts.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    set_seed(agent_cfg['seed'])
    
    # Parse env args
    env_cfg.scene.num_envs = opts.num_envs if opts.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = opts.device if opts.device is not None else env_cfg.sim.device
    if opts.distributed:
        env_cfg.sim.device = f"cuda:{device}"
    
    # Parse agent args
    agent_cfg['headless'] = opts.headless
    if opts.max_iterations:
        agent_cfg["trainer"]["timesteps"] = opts.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    
    # Configure the ML framework into the global skrl variable
    if opts.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if opts.ml_framework == "jax" else "numpy"
    
    # Specify directory for logging experiments
    log_root_path = os.path.join("logs", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # Specify directory for logging runs
    txt_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(opts.algorithm, txt_date) # opts.ml_framework
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    
    # Set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    
    # Dump the configuration into log-directory
    log_dir = os.path.join(log_root_path, log_dir)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    
    # Move hydra logs to log directory
    hydra_dir = max(
        (datetime.strptime(f"{d.parent.name} {d.name}", "%Y-%m-%d %H-%M-%S"), d)
        for d in Path("outputs").glob("*/*") if d.is_dir()
    )[1]._str # os.path.join("outputs", *txt_date.split("_"))
    time.sleep(1)
    if os.path.isdir(hydra_dir):
        shutil.move(os.path.join(hydra_dir, "hydra.log"), os.path.join(hydra_dir, ".hydra", "hydra.log"))
        shutil.move(os.path.join(hydra_dir, ".hydra"), os.path.join(log_dir, ".hydra"))
        if len(os.listdir(hydra_dir)) == 0:
            shutil.rmtree(hydra_dir)
        if len(os.listdir(os.path.dirname(hydra_dir))) == 0:
            shutil.rmtree(os.path.dirname(hydra_dir))
        if len(os.listdir("outputs")) == 0:
            shutil.rmtree("outputs")
    
    return env_cfg, agent_cfg, log_dir

def play_cfg(opts: argparse.ArgumentParser, test: bool = False) -> tuple[dict, dict, str]:
    """
    Configure and prepare the environment and agent for evaluation or play mode.

    Args:
        opts (argparse.ArgumentParser): Command-line options.
        test (bool, optional): Whether to set the environment in test (evaluation) mode. Defaults to False.

    Returns:
        tuple: (Environment config, agent config, log directory path)
    """
    from skrl.utils import set_seed
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

    # Parse env args
    env_cfg = parse_env_cfg(
        opts.task, device=opts.device, num_envs=opts.num_envs, use_fabric=not opts.disable_fabric
    )
    
    # Parse agent args
    try:
        agent_cfg = load_cfg_from_registry(opts.task, f"skrl_{opts.algorithm}_cfg_entry_point")
    except ValueError:
        agent_cfg = load_cfg_from_registry(opts.task, "skrl_cfg_entry_point")
    agent_cfg['headless'] = opts.headless
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    # Set agent and env seed. Note: certain randomization occur in the env initialization, so we set the seed here
    agent_cfg["seed"] = opts.seed if opts.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    set_seed(agent_cfg['seed'])

    # Specify directory for logging experiments
    log_root_path = os.path.abspath(
        os.path.join("logs", agent_cfg["agent"]["experiment"]["directory"])
    )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # Get checkpoint
    if opts.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", opts.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif opts.checkpoint:
        resume_path = os.path.abspath(opts.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{opts.algorithm}_{opts.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    
    return env_cfg, agent_cfg, log_dir
