from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from typing import Optional, Tuple, Any


class Grid2DEnv(gym.Env):
    """A simple 2D grid environment where an agent must reach a goal."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        grid_size: int = 20,
        reward_weight: float = 1.0,
        penalty_weight: float = -0.01,
        max_steps: int = 60,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.reward_weight = reward_weight
        self.penalty_weight = penalty_weight
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = Discrete(4)  # right (0), up (1), left (2), down (3)
        self.observation_space = Dict({
            "agent": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "goal": Box(low=0, high=1, shape=(2,), dtype=np.float32),
        })
        
        # State variables
        self.agent_pos = np.zeros(2, dtype=np.int32)
        self.goal_pos = np.zeros(2, dtype=np.int32)
        self.steps = 0
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_penalty = 0.0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Generate random positions for agent and goal
        self.agent_pos, self.goal_pos = self._random_positions()
        self.steps = 0
        
        # Reset episode tracking
        info = {
            "episode_reward": self.episode_reward,
            "episode_penalty": self.episode_penalty,
        }
        self.episode_reward = 0.0
        self.episode_penalty = 0.0
        
        observation = self._get_observation()
        return observation, info
    
    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """Execute one time step within the environment."""
        # Apply action
        self._apply_action(action)
        self.steps += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Check termination conditions
        terminated = self._is_goal_reached()
        truncated = self.steps >= self.max_steps
        
        # Calculate reward
        reward = self._calculate_reward(terminated)
        
        # Prepare info
        info = {}
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.steps,
            }
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: int) -> None:
        """Apply the action to move the agent."""
        # Calculate movement deltas
        dx = int(action == 0) - int(action == 2)  # +1 if right, -1 if left
        dy = int(action == 1) - int(action == 3)  # +1 if down, -1 if up
        
        # Update position
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        
        # Clamp to grid boundaries
        new_x = np.clip(new_x, 0, self.grid_size - 1)
        new_y = np.clip(new_y, 0, self.grid_size - 1)
        
        self.agent_pos[0] = new_x
        self.agent_pos[1] = new_y
    
    def _get_observation(self) -> dict:
        """Get the current observation."""
        return {
            "agent": self.agent_pos.astype(np.float32) / (self.grid_size - 1),
            "goal": self.goal_pos.astype(np.float32) / (self.grid_size - 1),
        }
    
    def _is_goal_reached(self) -> bool:
        """Check if the agent has reached the goal."""
        return np.array_equal(self.agent_pos, self.goal_pos)
    
    def _calculate_reward(self, terminated: bool) -> float:
        """Calculate the reward for the current step."""
        if terminated:
            reward = self.reward_weight
            penalty = 0.0
        else:
            reward = 0.0
            penalty = self.penalty_weight
        
        # Track episode totals
        self.episode_reward += reward
        self.episode_penalty += penalty
        
        return reward + penalty
    
    def _random_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random unique positions for agent and goal."""
        # Sample 2 unique positions
        all_positions = np.arange(self.grid_size * self.grid_size)
        selected = self.np_random.choice(all_positions, size=2, replace=False)
        
        # Convert to 2D coordinates
        agent_x, agent_y = selected[0] // self.grid_size, selected[0] % self.grid_size
        goal_x, goal_y = selected[1] // self.grid_size, selected[1] % self.grid_size
        
        agent_pos = np.array([agent_x, agent_y], dtype=np.int32)
        goal_pos = np.array([goal_x, goal_y], dtype=np.int32)
        
        return agent_pos, goal_pos
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render the environment in text mode."""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        grid[self.goal_pos[1], self.goal_pos[0]] = "G"
        grid[self.agent_pos[1], self.agent_pos[0]] = "A"
        
        print(f"\nStep: {self.steps}/{self.max_steps}")
        print("+" + "-" * (self.grid_size * 2 - 1) + "+")
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("+" + "-" * (self.grid_size * 2 - 1) + "+")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render the environment as RGB array."""
        # Create a simple RGB visualization
        cell_size = 20
        img = np.ones((self.grid_size * cell_size, self.grid_size * cell_size, 3), dtype=np.uint8) * 255
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            img[i * cell_size, :] = 200
            img[:, i * cell_size] = 200
        
        # Draw goal (green)
        gx, gy = self.goal_pos
        img[gx*cell_size:(gx+1)*cell_size, gy*cell_size:(gy+1)*cell_size] = [0, 255, 0]
        
        # Draw agent (blue)
        ax, ay = self.agent_pos
        img[ax*cell_size:(ax+1)*cell_size, ay*cell_size:(ay+1)*cell_size] = [0, 0, 255]
        
        return img
    
    def close(self):
        """Clean up resources."""
        pass

# Example usage
if __name__ == "__main__":
    env = Grid2DEnv(grid_size=10, max_steps=50, render_mode="human")
    
    observation, info = env.reset(seed=42)
    print(f"Initial observation: {observation}")
    
    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
        
        if terminated or truncated:
            observation, info = env.reset()
            print("Environment reset!")
    
    env.close()
