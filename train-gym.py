import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.runner.torch import Runner

from utils import load_runner
from nets import load_model
import envs.register

def main():

    # load a vectorized environment
    env = gym.make_vec("grid2d-gym-v0", num_envs=10, vectorization_mode="async")

    # wrap the environment
    env = wrap_env(env)
    
    # Load model
    model = load_model('mlp', env)
    
    # Get skrl runner
    agent_cfg = Runner.load_cfg_from_yaml("envs/agents/skrl_cfg_grid2d_gym.yaml")
    agent_cfg['headless'] = True
    runner = load_runner(
        env,
        agent_cfg,
        model=model,
        use_runner=False,
        ml_framework='torch',
        checkpoint='',
        train=True
    )

    # Run training
    runner.run()

    # Close simulator
    env.close()
    print("Finished training!")

if __name__ == "__main__":
    main()
