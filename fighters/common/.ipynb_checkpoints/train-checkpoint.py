from typing import Any
import time
import os
import gc

import torch
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from gymnasium import Env

from .constants import TB_DIR, OPT_DIR



def get_hyperparam_combos(params_table, cur_params={}, keys=None):
    if keys is None:
        keys = list(params_table.keys())
    if len(keys) == 0:
        return [cur_params]

    model_setups = []
    key = keys[0]
    for value in params_table[key]:
        new_params = {**cur_params, key: value}
        model_setups.extend(get_hyperparam_combos(params_table, new_params, keys[1:]))

    return model_setups


def train_model(model_class, env: Env, model_options: dict[str, Any], total_timesteps: int = 25000, n_eval_episodes=25, tb_log_name="model", log_interval=1, verbose=0, device="auto"):
    env.reset()
    start_time = time.time()

    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    model = model_class(env=env, verbose=verbose, device=device, tensorboard_log=TB_DIR, **model_options)
    print("Learning...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, log_interval=log_interval, progress_bar=True)

    learn_end_time = time.time()
    learn_elapsed_time = learn_end_time - start_time

    print(f"Learn time: {learn_elapsed_time/60:.3f} minutes")

    arch, labels = evaluate_model(model, env, model_options, n_eval_episodes, learn_elapsed_time)

    save_model(model, labels)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return arch


def evaluate_model(model, env, model_options, n_eval_episodes, learn_elapsed_time):
    print(f"Evaluating over {n_eval_episodes} episodes...")
    start_time = time.time()

    ep_rewards, ep_lengths = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards = True, deterministic = False)
    reward_mean = np.mean(ep_rewards)
    reward_stdev = np.std(ep_rewards)
    length_mean = np.mean(ep_lengths)
    length_stdev = np.std(ep_lengths)

    eval_end_time = time.time()
    eval_elapsed_time = eval_end_time - start_time
    total_elapsed_time = learn_elapsed_time + eval_elapsed_time

    option_values = [model_options[key] for key in model_options]
    arch = option_values + [ep_rewards, ep_lengths, reward_mean, length_mean, learn_elapsed_time, eval_elapsed_time]
    labels = [str(value) for value in option_values] + [str(reward_mean), str(length_mean)]

    print(f"Evaluation time: {eval_elapsed_time/60:.3f} minutes")
    print(f"Mean episode reward: {reward_mean}, Standard deviation: {reward_stdev}")
    print(f"Mean episode length: {length_mean}, Standard deviation: {length_stdev}")
    print(f'finished architecture {arch} at {total_elapsed_time/60:.3f} minutes.')

    return arch, labels


def save_model(model, labels):
    SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format("_".join(labels)))
    model.save(SAVE_PATH)
