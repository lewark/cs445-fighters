from .fighter_envs import StreetFighter, make_env, train_model

def search(params_table, cur_params={}, keys=None):
    if keys is None:
        keys = list(params_table.keys())
    if len(keys) == 0:
        return [cur_params]

    model_setups = []
    key = keys[0]
    for value in params_table[key]:
        new_params = {**cur_params, key: value}
        model_setups.extend(search(params_table, new_params, keys[1:]))

    return model_setups


if __name__ == "__main__":
    from stable_baselines3 import A2C

    params = {
        "policy": ['CnnPolicy'],
#        "policy": ['CnnPolicy', 'MlpPolicy'],
#        "learning_rate": [0.1, 0.01, 0.001],
#        "n_steps": [1024], #, 32, 64, 128, 256]
#        "use_rms_prop": [True, False],
#        "gamma": [0.99, 0.5],
#        "gae_lambda": [1, 0.5, 0],
        #"use_sde": [False, True]
    }

    n_procs = 8
    n_stack = 4

    model_setups = search(params)
    print("Training", len(model_setups), "models")

    env = make_env(StreetFighter, n_procs, n_stack, render_mode=None, random_delay=30)
    for model_options in model_setups:
        print(model_options)
        train_model(A2C, env, model_options, total_timesteps=500000, log_interval=1)
    env.close()
