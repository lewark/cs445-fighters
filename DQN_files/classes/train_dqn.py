from typing import Any, Callable
import numpy as np
import time
import os

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from gymnasium import Env

from classes.constants import TB_DIR, OPT_DIR
from classes.utils import cleanup_device, get_device
from classes.evaluate import evaluate, get_eval_stats
from classes.dqn_envs import make_dqn_env


def train_model_dqn_2(model_class, base_env: Env, env_func: Callable, model_options: dict[str, Any], total_timesteps: int = 25000, n_eval_episodes=25, tb_log_name="DQN", log_interval=4, verbose=0, device="auto"):
    start_time = time.time()
    env = env_func(base_env, render_mode = None)

    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    model = model_class(env=env, verbose=verbose, device=device, tensorboard_log=TB_DIR, **model_options)
    print("Learning...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, log_interval=log_interval, progress_bar=True)
    env.close()

    learn_end_time = time.time()
    learn_elapsed_time = learn_end_time - start_time

    print(f"Learn time: {learn_elapsed_time/60:.3f} minutes")
    eval_env = env_func(base_env, render_mode = None)

    arch, labels = evaluate_model(model, env, model_options, n_eval_episodes, learn_elapsed_time)
    eval_env.close()

    save_model(model, labels)

    del model
    cleanup_device()

    return arch


def evaluate_model(model, env, model_options, n_eval_episodes, learn_elapsed_time):
    print(f"Evaluating over {n_eval_episodes} episodes...")
    start_time = time.time()

    ep_rewards, ep_stds = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards = True, deterministic = False)
    reward_mean = np.mean(np.array(ep_rewards))
    std_mean = np.mean(np.array(ep_stds))
    reward_sum = np.sum(np.array(ep_rewards))

    eval_end_time = time.time()
    eval_elapsed_time = eval_end_time - start_time
    total_elapsed_time = learn_elapsed_time + eval_elapsed_time

    option_values = [model_options[key] for key in model_options]
    arch = option_values + [ep_rewards, ep_stds, reward_mean, std_mean, reward_sum, learn_elapsed_time, eval_elapsed_time]
    labels = [str(value) for value in option_values] + [str(reward_mean), str(std_mean), str(reward_sum)]

    print(f"Evaluation time: {eval_elapsed_time/60:.3f} minutes")
    print(f'finished architecture {arch} at {total_elapsed_time/60:.3f} minutes.')

    return arch, labels


def save_model(model, labels):
    SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format("_".join(labels)))
    model.save(SAVE_PATH)


def train_model_dqn(env_class, 
                    model_options: dict[str, Any], 
                    n_procs: int = 0, 
                    n_stack: int = 4, 
                    total_timesteps: int = 10_000,
                    n_eval_episodes = 5,
                    render_mode = None,
                    ev_func = 'evaluate_policy',
                    verbose = 0,
                    p_bar = False,
                    plot = False):
    
    env = None
    eval_env = None
    reward_mean = None
    reward_std = None
    reward_sum = None
    info = None
    device = 'cpu'

    try:
        start_time = time.time()

        # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
        device = get_device()

        env = make_dqn_env(env_class, render_mode=render_mode, n_procs=n_procs, n_stack=n_stack)
        model = DQN(env=env, verbose=verbose, device=device, **model_options)

        print("Learning...")
        model.learn(total_timesteps=total_timesteps, progress_bar=p_bar)
        env.close()

        print("Evaluating...")
        eval_env = make_dqn_env(env_class, render_mode=render_mode, n_procs=n_procs, n_stack=n_stack)

        if ev_func == 'evaluate_policy':
            ep_rewards, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, return_episode_rewards = True, deterministic=False)
            reward_mean, reward_std, reward_sum = get_eval_stats(np.array(ep_rewards))

        elif plot:
            reward_mean, reward_std, reward_sum, ep_info = evaluate(model, eval_env, n_eval_episodes, deterministic=False, extra=True)
        
        else:
            reward_mean, reward_std, reward_sum = evaluate(model, eval_env, n_eval_episodes, deterministic=False, return_episode_rewards = True)
        
        del model
        
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
            
    elapsed = (time.time() - start_time) /60
    cleanup_device()
    
    elapsed = round(elapsed, 4)

    if plot:
        return reward_mean, reward_std, reward_sum, elapsed, info
    else:
        return reward_mean, reward_std, reward_sum, elapsed