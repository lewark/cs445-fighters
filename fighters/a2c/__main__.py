from ..common.fighter_envs import make_env
from .a2c_envs import A2CStreetFighter, RewardFunction
from ..common.train import train_model, get_hyperparam_combos
from stable_baselines3 import A2C, PPO


if __name__ == "__main__":
    params = {
        "policy": ['CnnPolicy', 'MlpPolicy'],
        "learning_rate": [0.000001, 0.00001, 0.0001], #, 0.001, 0.01],
        "n_steps": [5], #, 16]
#        "use_rms_prop": [True, False],
#        "gamma": [0.99, 0.5],
#        "gae_lambda": [1, 0.5, 0],
        #"use_sde": [False, True]
    }

    n_procs = 8
    n_stack = 4
    total_timesteps = 500000

    model_setups = get_hyperparam_combos(params)
    print("Training", len(model_setups), "models")

    env = make_env(lambda: A2CStreetFighter(RewardFunction(use_distance=False), random_delay=0, render_mode=None), n_procs, n_stack)
    for model_options in model_setups:
        print(model_options)
        learning_rate = model_options["learning_rate"]
        n_steps = model_options["n_steps"]
        label = f"A2C_{learning_rate}_{n_steps}_{total_timesteps}"

        train_model(A2C, env, model_options, total_timesteps=total_timesteps, log_interval=100, n_eval_episodes=25, tb_log_name=label, device="auto")
    env.close()
