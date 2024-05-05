from matplotlib import pylab as plt
from tqdm import tqdm
import numpy as np
import os

from gymnasium import Env


from stable_baselines3.common.base_class import BaseAlgorithm

# *Based on https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb
#  modified version.
def evaluate(
    model: BaseAlgorithm,
    eval_env,
    n_eval_episodes: int = 5,
    deterministic: bool = False,
    upper_n_step_bound: int = 10_000_000,
    p_bar_off: bool = True,
    info_list = None     # allows for game statistics from the info dict to be returned.
) -> dict:
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param n_eval_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :param upper_n_step_bound: max number of steps that a model can run in a single episode before returning.
                                    This is here specifically to prevent cases where a model may complete an entire game
                                    and then be stuck in an infinite loop during an evaluation episode.
    :param: p_bar_off: if set to false, it will turn on a tqdm progress bar for the evaluation episodes.

    Returns: The statistics for each episode are recorded and returned using a dict 'ep_recorder', where
    the episode number is the key. The statistics recorded are 'total episode steps: int, the running
    sum of each episode rewards: list(int, int, int ...), and the rewards received by the agent at each
    timestep: list(int, int, int ...). In addition, any extra game information you want to get back from
    the game during evaluation can be obtained from the info object by providing the method with an info_list

    Example: info_list = ['matches_won', 'score']
    eval_results = evaluate(model=model, env=env, n_eval_episodes=30, info_list= info_lst)

    results = get_eval_stats(eval_results, get_episode_info=True)

    print(results)

    {'mean episode rewards': 4.25,
     'std episode rewards': 14.078,
     'total trial rewards': 85.0,
     'mean episode steps': 1664.55,
     'mean ep matches_won': 0.35,
     'mean ep score': 9565.0}

    """

    ep_recorder = {}
    for ep_num in tqdm(range(n_eval_episodes), disable=p_bar_off):
        done = False
        obs = eval_env.reset()

        ep_total_num_steps = 0
        ep_running_sum = 0
        ep_rsum_per_step = []        # For recording running sum of rewards per step.
        ep_rewards_per_step = []     # For recording the rewards per step.
        ep_info = None               # For recording extra info, such as 'rounds won' from the info dict
        
        while not done:
            
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = eval_env.step(action)
            ep_rewards_per_step.append(float(np.copy(reward[0])))
            ep_total_num_steps  += 1

            if ep_total_num_steps > upper_n_step_bound:
                print(f'episode exceeded {upper_n_step_bound} steps!')
                eval_env.close()
                return ep_recorder
            
            if info_list is not None and done:
                ep_info = [{k:info[0][k]} for k in info_list]
                
            
            ep_running_sum  += float(np.copy(reward[0]))
            ep_rsum_per_step.append(ep_running_sum)

        ep_recorder[ep_num] = [ep_total_num_steps, ep_rsum_per_step, ep_rewards_per_step, ep_info]
    
    eval_env.close()
    return ep_recorder


def evaluate_baseline(
    eval_env: Env,
    n_eval_episodes: int = 10,
    discrete: bool = False,  # for if you are using a discrete environment.
    info_list = None         # allows for game statistics from the info dict to be returned.
) -> dict:
    """
    Evaluate baseline RL agent for `num_episodes' without a model (random agent)

    The statistics for each episode are recorded and returned using a dict 'ep_recorder', where
    the episode number is the key. The statistics recorded are 'total episode steps: int, the running
    sum of each episode rewards: list(int, int, int ...), and the rewards received by the agent at each
    timestep: list(int, int, int ...). In addition, any extra game information you want to get back from
    the game during evaluation can be obtained from the info object by providing the method with an info_list

    Example: info_list = ['matches_won', 'score']
    eval_results = evaluate_baseline(env=env, n_eval_episodes=30, info_list= info_lst)

    results = get_eval_stats(eval_results, get_episode_info=True)

    print(results)

    {'mean episode rewards': 4.25,
     'std episode rewards': 14.078,
     'total trial rewards': 85.0,
     'mean episode steps': 1664.55,
     'mean ep matches_won': 0.35,
     'mean ep score': 9565.0}
   
    """
    
    # change log -- making this use a single separate eval env
    # instead of the model's env.
    ep_recorder = {}

    for ep_num in tqdm(range(n_eval_episodes)):
        done = False
        obs = eval_env.reset()

        ep_total_num_steps = 0
        ep_running_sum = 0
        ep_rsum_per_step = []        # For recording running sum of rewards per step.
        ep_rewards_per_step = []     # For recording the rewards per step.
        ep_info = None               # For recording extra info, such as 'rounds won' from the info dict
        
        while not done:
            
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            
            if discrete:
                obs, reward, done, info = eval_env.step(eval_env.action_space.sample().reshape(1,))
                ep_rewards_per_step.append(float(np.copy(reward[0])))

                ep_total_num_steps  += 1

                if ep_total_num_steps > 1_000_000:
                    print('episode exceeded 10 million steps!')
                    eval_env.close()
                    return ep_recorder
                
                if info_list is not None and done:
                    ep_info = [{k:info[0][k]} for k in info_list]
                
                ep_running_sum  += float(np.copy(reward[0]))
                ep_rsum_per_step.append(ep_running_sum)
                
            else:
                obs, reward, done, info = eval_env.step(eval_env.action_space.sample())
                ep_rewards_per_step.append(float(np.copy(reward[0])))

                ep_total_num_steps  += 1

                if ep_total_num_steps > 1_000_000:
                    print('episode exceeded 10 million steps!')
                    eval_env.close()
                    return ep_recorder
                
                if info_list is not None and done:
                    ep_info = [{k:info[0][k]} for k in info_list]
                
                ep_running_sum  += float(np.copy(reward[0]))
                ep_rsum_per_step.append(ep_running_sum)

        ep_recorder[ep_num] = [ep_total_num_steps, ep_rsum_per_step, ep_rewards_per_step, ep_info]
    
    eval_env.close()
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


def get_eval_stats(ep_recorder: dict, return_values: bool = False, get_episode_info: bool = True):
    all_ep_rewards = []
    all_ep_total_steps = []
    all_ep_info = []
    all_ep_reward_sum = 0

    for episode in ep_recorder:
        total_steps, rsums_per_step, _, ep_info = ep_recorder[episode]

        # The last value of the running sum = total of episode rewards.
        all_ep_rewards.append(rsums_per_step[-1])
        all_ep_total_steps.append(total_steps)
        all_ep_reward_sum += rsums_per_step[-1]
        all_ep_info.append(ep_info)

    all_ep_rewards_mean = np.mean(all_ep_rewards)
    all_ep_std = np.std(all_ep_rewards)
    all_ep_steps_mean = np.mean(all_ep_total_steps)

    results_dict = {'mean episode rewards': round(all_ep_rewards_mean,3),
                    'std episode rewards': round(all_ep_std,3),
                    'total trial rewards': round(all_ep_reward_sum,3),
                    'mean episode steps': round(all_ep_steps_mean,3)}

    if return_values and get_episode_info:
        return all_ep_rewards_mean, all_ep_std, all_ep_reward_sum, all_ep_steps_mean, ep_info

    elif return_values:
        return all_ep_rewards_mean, all_ep_std, all_ep_reward_sum, all_ep_steps_mean
    
    elif get_episode_info and all_ep_info[0] is not None:
        values = {}
        
        for ep_info in all_ep_info:
            for i, dict_obj in enumerate(ep_info):
                key = [*dict_obj.keys()]
                val = [*dict_obj.values()]
                if key[0] not in values:
                    values[key[0]] = []
                values[key[0]].append(val[0])

        for keys in values:
            values[keys] = np.mean(values[keys])

        for k in values:
            results_dict[f'mean ep {k}'] = round(values[k], 3)
                
        return results_dict
    else:
        return results_dict