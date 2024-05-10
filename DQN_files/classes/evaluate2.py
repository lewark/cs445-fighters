from matplotlib import pylab as plt
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import os

from gymnasium import Env
from classes.utils import add_to_recorder


from stable_baselines3.common.base_class import BaseAlgorithm
# *Based on https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb
#  modified version.

def _baseline_predict(eval_env, discrete):
    action = None
    if discrete:
        action = eval_env.action_space.sample().reshape(1,)
    else:
        action = eval_env.step(eval_env.action_space.sample())
    return action
    

def evaluate(
    model: BaseAlgorithm,
    eval_env,
    n_eval_episodes: int = 25,
    deterministic: bool = False,
    return_episode_rewards: bool = False,
    progress_bar: bool = True,
    get_ep_info: bool = False,
    upper_n_step_bound: int = 10_000_000,
    discrete: bool = True
) -> dict:
    """
    Evaluate an RL agent for `num_episodes`.
    Can perform regular prediction if given a model, or baseline
    prediction of a random agent if model is set to None.

    returns: mean, std of episode rewards, and mean episode steps if return_episode_rewards is False
    otherwise it will return a dict with recorded information from each episode, where the key of the
    dict is the episode number.

    the values of the dict will be:
    ep_total_num_steps, ep_running_sum_per_step, ep_rewards_per_step, ep_info]

    where ep_info will be a dict congaing the information in the episode info object per episode step
    if get_ep_info is set to true.

    """
    obs = eval_env.reset()
    all_episode_rewards = []
    all_episode_total_steps = []
    all_end_ep_info = {}

    ep_recorder = {}
    for ep_num in tqdm(range(n_eval_episodes), disable = (not progress_bar)):
        done = False
        obs = eval_env.reset()

        ep_total_num_steps = 0
        ep_running_sum = 0
        ep_rsum_per_step = []        # For recording running sum of rewards per step.
        ep_rewards_per_step = []     # For recording the rewards per step.
        ep_info = {}

        # player_rounds_won = 0
        # player_rounds_won_per_step = 0
        player_KOs = 0
        player_OKs_per_step = 0
        # enemy_rounds_won = 0
        # enemy_rounds_won_per_step = 0
        enemy_KOs = 0
        enemy_OKs_per_step = 0
        
        while not done:
            action = None
            if model is None:
                action = _baseline_predict(eval_env, discrete)
            else:  
                action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = eval_env.step(action)
            ep_rewards_per_step.append(int(np.copy(reward[0])))
            ep_total_num_steps  += 1

            if ep_total_num_steps > upper_n_step_bound:
                print(f'episode exceeded {upper_n_step_bound} steps!')
                eval_env.close()
                return ep_recorder
                
            ep_running_sum  += int(np.copy(reward[0]))
            ep_rsum_per_step.append(ep_running_sum)

            if get_ep_info:
                info_copy = deepcopy(info[0])
                add_to_recorder(ep_info, info_copy)

                # if info_copy['matches_won'] > player_rounds_won_per_step:
                #     player_rounds_won_per_step = info_copy['matches_won']
                #     player_rounds_won += 1
                
                if info_copy['health'] == -1:
                    player_OKs_per_step =+ 1

                if player_OKs_per_step > 0 and info_copy['health'] != -1:
                    player_OKs_per_step = 0
                    player_KOs += 1
                
                if info_copy['enemy_health'] == -1:
                    enemy_OKs_per_step =+ 1

                if enemy_OKs_per_step > 0 and info_copy['enemy_health'] != -1:
                    enemy_OKs_per_step = 0
                    enemy_KOs += 1
                    
                # if info_copy['enemy_matches_won'] > enemy_rounds_won_per_step:
                #     enemy_rounds_won_per_step = info_copy['enemy_matches_won']
                #     enemy_rounds_won += 1
                    
                if done:
                    mean_player_health = np.mean(ep_info['health'])
                    mean_enemy_health = np.mean(ep_info['enemy_health'])
                    
                    ep_info_stats = {}
                    ep_info_stats['episode'] = (ep_num + 1)
                    ep_info_stats['Player rounds won'] = enemy_KOs
                    ep_info_stats['Mean player health'] = round(mean_player_health, 2)
                    ep_info_stats['Enemy rounds won'] = player_KOs
                    ep_info_stats['Mean enemy health'] = round(mean_enemy_health, 2)
                    add_to_recorder(all_end_ep_info, ep_info_stats)


        all_episode_total_steps.append(ep_total_num_steps)
        all_episode_rewards.append(ep_running_sum)
        ep_recorder[ep_num] = [ep_total_num_steps, ep_rsum_per_step, ep_rewards_per_step, 'ep_info here is not implemented']
    
    eval_env.close()

    if return_episode_rewards:
        mean = np.mean(all_episode_rewards)
        std = np.std(all_episode_rewards)
        total = np.sum(all_episode_rewards)
        mean_steps = np.mean(all_episode_total_steps)

        return mean, std, total, mean_steps, all_end_ep_info
    else:
        return ep_recorder


def eval_plotter(ep_recorder: dict, trial = None, legend = True, plot_each = False, show_all = False, title = 'title'):

    for k in ep_recorder.keys():
        steps, rsum, all_rewards, _ep_info = ep_recorder[k]
        all_steps = [i for i in range(steps)]

        plt.plot(all_steps, rsum, label = f'episode: {k}')
        plt.title(title)
        plt.ylabel('Cumulative Rewards')
        plt.xlabel('Steps')
        if legend:
            plt.legend()
        if plot_each:
            plt.show()

    plt.show()

    if show_all:
        alpha = 1.0
        n_episodes = len(ep_recorder)
        alpha_offset = (0.8/(n_episodes))
        
        for k in ep_recorder.keys():
            steps, rsum, all_rewards, _ep_info = ep_recorder[k]
            all_steps = [i for i in range(steps)]

            plt.plot(all_steps, all_rewards, '.', label = f'episode: {k}', alpha = alpha)
            plt.title(f'Trial {trial} Rewards per Step: All Episodes')
            plt.ylabel('Rewards')
            plt.xlabel('Steps')
            if legend:
                plt.legend()
            if plot_each:
                plt.show()
            alpha -= alpha_offset
            if alpha <= 0.15:
                alpha = 0.15
                
    plt.show()


def calculate_grid_size(num_items, num_columns = 4):
    if num_items <= num_columns:
        num_columns = num_items
    num_rows = int(np.ceil(num_items / num_columns))
    return num_rows, num_columns


def eval_subplots(num_plots, 
                  image_list, 
                  num_cols = 3, 
                  legend = False,
                  figsize = (5, 3),
                  title = 'title'):
    trial = 1

    num_plots = min(num_plots, len(image_list))
    row, col = calculate_grid_size(num_plots, num_columns=num_cols)
    
    plt.rcParams['figure.figsize'] = figsize
    fig, axes = plt.subplots(row, col)
    
    i = 0
    for ax in axes.flat:
        if i < num_plots:
            image = image_list[i]

            for ep in image:
                steps, rsum, _ = image[ep]
                all_steps = [i for i in range(steps)]
                x, y  = all_steps, rsum
                ax.plot(x, y, label = ep)
                    
                # ax.axis('off')
                ax.set_title(f'Trial {trial}: {title}')
                ax.set_xlabel('Steps')
                ax.set_ylabel('Cumulatve Rewards')
                if legend:
                    ax.legend()
        else:
            ax.axis('off')
        
        i += 1
        trial += 1
    plt.tight_layout()
    plt.title(title)
    plt.show()


def eval_plots_all(image_list, all_rewards=False, title = 'title'):
    for image in image_list:
        for ep in image:
            steps, rsum, rewards = image[ep]
            all_steps = [i for i in range(steps)]

            if not all_rewards:
                plt.plot(all_steps, rsum)
                plt.ylabel('Cumulative Rewards')
                plt.xlabel('Steps')
                plt.title(f'{title}')
            else:
                plt.plot(all_steps, rewards)
                plt.ylabel('Rewards')
                plt.xlabel('Steps')
                plt.title(f'{title}')
    plt.title(title)
    plt.show()


def save_episode_plots(ep_recorder: dict, save_dir, trial, legend=True, getting_running_sum = True, title = 'title'):
    for k in ep_recorder.keys():
        if getting_running_sum:
            steps, rsum, all_rewards = ep_recorder[k]
            all_steps = [i for i in range(steps)]
            plt.plot(all_steps, rsum, label=f'episode: {k}')
            plt.title(title)
            plt.ylabel('Cumulative Rewards')
            plt.xlabel('Steps')
            if legend:
                plt.legend()
        else:
            steps, rsum, all_rewards = ep_recorder[k]
            all_steps = [i for i in range(steps)]
            plt.plot(steps, all_rewards, label=f'episode: {k}')
            plt.title(title)
            plt.ylabel('Cumulative Rewards')
            plt.xlabel('Steps')
            if legend:
                plt.legend()

    save_path = os.path.join(save_dir, f'trial_{trial}.svg')
    try:
        plt.savefig(save_path, )
        print(f'saving image to {save_path}')
    except Exception as e:
        print(f'image could not be saved to {save_path}')
        print(e)
        plt.close('all')
    finally:
        plt.close('all')
        return save_path


def get_eval_stats(ep_recorder: dict, return_values: bool = False):
    all_ep_total_rewards = []
    all_ep_total_steps = []
    all_ep_rewards_per_step = []
    all_ep_reward_sum = 0

    for episode in ep_recorder:
        total_steps, rsums_per_step, rewards_per_step, _ = ep_recorder[episode]

        # The last value of the running sum = total of episode rewards.
        all_ep_total_rewards.append(rsums_per_step[-1])
        all_ep_total_steps.append(total_steps)
        all_ep_reward_sum += rsums_per_step[-1]
        all_ep_rewards_per_step.append(rewards_per_step)

    all_ep_rewards_mean = np.mean(all_ep_total_rewards)
    all_ep_std = np.std(all_ep_total_rewards)
    all_ep_steps_mean = np.mean(all_ep_total_steps)
    # all_ep_mean_reward_per_step = np.mean(all_ep_rewards_per_step)

    results_dict = {'Mean total episode rewards': round(all_ep_rewards_mean,3),
                    'Std total episode rewards': round(all_ep_std,3),
                    'Reward sum all episodes': round(all_ep_reward_sum,3),
                    'Mean episode steps': round(all_ep_steps_mean,3)}

    if return_values:
        return all_ep_total_rewards, all_ep_total_steps, all_ep_rewards_per_step, all_ep_reward_sum
    else:
        return results_dict
    
def get_ep_info(ep_recorder):
    ep_info_list = []
    for ep in ep_recorder:
        steps, _, _, ep_info = ep_recorder[ep]

        ep_info_list.append((steps, ep_info))
    return ep_info_list


# def get_ep_info_stats(ep_recorder:dict, info_attributes: list, verbose: int = 0):
#     ep_info_summery = {}
#     ep_list = get_ep_info(ep_recorder)

#     for key in info_attributes:
        
#         ep_value_list = [(ep_info[0], ep_info[1][key]) for ep_info in ep_list]
#         try:
#             mean = np.mean([np.mean(sum(val_list[1]) / val_list[0]) for val_list in ep_value_list])
#             ep_info_summery[f'Mean episode: {key}:'] = round(mean, 3)
#         except Exception as e:
#             if verbose > 0:
#                 print(f'could not generate mean for {key}')
#             if verbose > 1:
#                 print(e.with_traceback())
#         try:
#             sum_vals = sum([sum(sum(val_list[1]) / val_list[0]) for val_list in ep_value_list])
#             ep_info_summery[f'Total episode: {key}:'] = round(sum_vals, 3)
#         except Exception as e:
#             if verbose > 0:
#                 print(f'could not generate a sum for {key}')
#             if verbose > 1:
#                 print(e.with_traceback())
#         try:
#             std = np.std([np.std(sum(i[1]) / i[0]) for i in ep_value_list])
#             ep_info_summery[f'Mean episode std: {key} :'] = round(std, 3)
#         except Exception as e:
#             if verbose > 0:
#                 print(f'could not generate a sum for {key}')
#             if verbose > 1:
#                 print(e.with_traceback())
    
#     return ep_info_summery


