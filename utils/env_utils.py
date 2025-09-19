import os
import torch
import gymnasium as gym

def take_action(env, runner, obs):
    
    # Take action
    with torch.inference_mode():
        
        # Agent action
        # outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        outputs = runner.agent.policy.compute({"states": runner.agent._state_preprocessor(obs)}, role="policy")
        
        # Multi-agent (deterministic) actions
        if hasattr(env, "possible_agents"):
            actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
        
        # Single-agent (deterministic) actions
        else:
            # actions = outputs[-1].get("mean_actions", outputs[0])
            actions = outputs[0].softmax(dim=-1).argmax(dim=-1)
    
    return actions

def get_env(opts, env_cfg, log_dir, train=False):
    from isaaclab.utils.dict import print_dict
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

    # Create isaac env
    env = gym.make(opts.task, cfg=env_cfg, render_mode="rgb_array" if opts.video else None)

    # Convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and opts.algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Wrap for video recording
    if opts.video:
        txt = "train" if train else "play"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", txt),
            "step_trigger": lambda step: (step % opts.video_interval == 0) if train else (step == 0),
            "video_length": opts.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during execution.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=opts.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`
    return env
