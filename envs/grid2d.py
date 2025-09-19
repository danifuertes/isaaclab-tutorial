from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from gymnasium.spaces import Dict, Box, Discrete

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class Grid2DEnvCfg(DirectRLEnvCfg):
    
    # Env parameter
    grid_size           : int   = 20            # Size of the grid (grid_size x grid_size)
    
    # Spaces
    state_space         : int  = 0                              # No state space (used for asymmetric actor-critic)
    action_space        : Box  = Discrete(4)                    # Action space: right (0), up (1), left (2), down (3)
    observation_space   : Dict = Dict({                         # Observation space
        "agent": Box(low=0, high=1, shape=(2,), dtype=float),   # -> Agent position
        "goal": Box(low=0, high=1, shape=(2,), dtype=float),    # -> Goal position
    })

    # Simulation
    decimation          : int   = 1             # Number of physics steps per decision step
    episode_length_s    : float = 1.0           # Episode length (in seconds)
    dt                  : float = 1 / 60        # Simulation time step (in seconds). Basically, inverse of frames per second
    env_spacing         : float = grid_size + 2 # Spacing between parallel envs
    num_envs            : int   = 1024          # Number of environments to simulate in parallel
    
    # Scene
    sim                 : SimulationCfg         = SimulationCfg(dt=dt, render_interval=decimation)                  # Simulation
    scene               : InteractiveSceneCfg   = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing)   # Interactive scene
    
    # Rewards
    reward_weight       : float = 1.0           # Reward for success
    penalty_weight      : float = -0.01         # Penalty for failure

class Grid2DEnv(DirectRLEnv):
    cfg: Grid2DEnvCfg

    def __init__(self, cfg: Grid2DEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Variables
        self.agent = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self.goal = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self.steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # Log info
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "reward",
                "penalty",
            ]
        }

    def _setup_scene(self):
        # Clone env setup
        self.scene.clone_environments(copy_from_source=False)
    
    def get_ids(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        num_envs = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]
        return env_ids, num_envs, env_origins

    def _reset_idx(self, env_ids: Sequence[int] | None):
        env_ids, num_envs, env_origins = self.get_ids(env_ids)
        super()._reset_idx(env_ids)
        
        # Set random positions
        agent, goal = self.random_positions(self.cfg.grid_size, num_envs, device=self.device)
        self.agent[env_ids] = agent
        self.goal[env_ids] = goal
        self.steps[env_ids] = 0

        # Reset log info
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

    def _get_observations(self) -> dict:
        # Observation
        observation = {
            'agent': self.agent.float() / (self.cfg.grid_size - 1),
            'goal': self.goal.float() / (self.cfg.grid_size - 1),
        }
        return {'policy': observation}

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().to(self.device).squeeze(dim=-1)

    def _apply_action(self):

        # Movement
        dx = (self.actions == 0).long() - (self.actions == 2).long()   # +1 if right, -1 if left
        dy = (self.actions == 1).long() - (self.actions == 3).long()   # +1 if down, -1 if up
        new_x = self.agent[:, 0] + dx
        new_y = self.agent[:, 1] + dy
        
        # Avoid out of bounds
        new_x = torch.clamp(new_x, 0, self.cfg.grid_size - 1)
        new_y = torch.clamp(new_y, 0, self.cfg.grid_size - 1)
        
        # Update
        self.agent[:, 0] = new_x
        self.agent[:, 1] = new_y
        self.steps += 1
        
    def _get_rewards(self) -> torch.Tensor:
        terminated, _ = self._get_dones()
        
        # Penalty for not reaching goal
        penalty = torch.zeros_like(terminated).float()
        penalty[~terminated] = -0.01

        # Collect rewards
        rewards = {
            'reward': self.cfg.reward_weight * terminated.float(),  #    +1 if goal is reached
            'penalty': self.cfg.penalty_weight * penalty,           # -0.01 if goal isn't reached
        }
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        # Return sum of rewards
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Terminated: Goal reached
        terminated = (self.agent == self.goal).all(dim=1)

        # Truncated: Timeout
        truncated = self.steps >= self.max_episode_length

        return terminated, truncated
    
    @staticmethod
    def random_positions(grid_size, num_envs, device='cpu'):

        # Sample 2 unique positions per env in parallel
        probs = torch.ones((num_envs, grid_size * grid_size), device=device)
        idx = torch.multinomial(probs, num_samples=2, replacement=False)
        
        # Reshape indices to 2D
        x1, y1 = idx[:, 0] // grid_size, idx[:, 0] % grid_size
        x2, y2 = idx[:, 1] // grid_size, idx[:, 1] % grid_size

        # Stack and return positions
        agent = torch.stack([x1, y1], dim=-1)
        goal = torch.stack([x2, y2], dim=-1)
        return agent, goal
