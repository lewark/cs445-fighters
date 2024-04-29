import fighter_envs

if __name__ == "__main__":
    from stable_baselines3 import A2C

    policies = ['CnnPolicy', 'MlpPolicy']
    n_learning_rate = [0.01, 0.001, 0.0001, 0.00001]
    batch_size = 64
    n_steps_values = [1024] #, 32, 64, 128, 256]
    n_procs = 8
    n_stack = 4

    model_setups = []
    for policy in policies:
        for learning_rate in n_learning_rate:
            for n_steps in n_steps_values:
                model_setups.append({"policy": policy, "learning_rate": learning_rate, "n_steps": n_steps})

    #model_setups=[{"policy":"CnnPolicy"}]
    env = fighter_envs.make_env(fighter_envs.StreetFighter, n_procs, n_stack, render_mode=None)
    for model_options in model_setups:
        print(model_options)
        fighter_envs.train_model(A2C, env, model_options, 1000000)
    env.close()
