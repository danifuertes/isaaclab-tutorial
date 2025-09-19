import gymnasium as gym

from . import agents

gym.register(
    id="grid2d-v0",
    entry_point="envs.grid2d:Grid2DEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "envs.grid2d:Grid2DEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg_grid2d.yaml",
    },
)
