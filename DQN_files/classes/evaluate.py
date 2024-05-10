import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

from stable_baselines3.common.base_class import BaseAlgorithm

# *Based on https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb
#  modified version.
def evaluate(
    
    model: BaseAlgorithm,
    eval_env,
    n_eval_episodes: int = 5,
    deterministic: bool = False
) -> dict:
    """
    Evaluate an RL agent for `num_episodes`

    ** Warning! : this class will only work properly when evaluating
    an model with a single environent ** .

    :param model: the RL Agent
    :param env: the gym Environment
    :param n_eval_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
   
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
        ep_rewards_per_step = []       # For recording the rewards per step.
        
        while not done:
            # _states are only useful when using LSTM policies
            # `deterministic` is to use deterministic actions
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, _info = eval_env.step(action)
            ep_rewards_per_step.append(int(reward))

            # print(reward)
            # print(type(reward))
            # print(action.shape)

            ep_total_num_steps  += 1
            ep_running_sum  += reward
            ep_rsum_per_step.append(ep_running_sum)

        """
            The statistics for each episode are recorded and returned using a dict 'ep_recorder', where
            the episide number is the key. The statistics recorded are 'total episode steps: int, the running
            sum of each episode rewards: list(int, int, int ...), and the rewards recieved by the agent at each
            timestep: list(int, int, int ...) 

        """

        ep_recorder[ep_num] = [ep_total_num_steps,  ep_rsum_per_step, ep_rewards_per_step]
    
    eval_env.close()
    return ep_recorder
    

def eval_plotter(ep_recorder: dict, trial = None, legend = True, plot_each = False, show_all = False):

    for k in ep_recorder.keys():
        steps, rsum, all_rewards = ep_recorder[k]
        all_steps = [i for i in range(steps)]

        plt.plot(all_steps, rsum, label = f'episode: {k}')
        plt.title(f'Trial {trial} Cumulative Rewards per Step: All Episodes')
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
            steps, rsum, all_rewards = ep_recorder[k]
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


def eval_subplots(num_plots, 
                  image_list, 
                  num_cols = 3, 
                  legend = False,
                  figsize = (5, 3),
                  trial = 1,
                  title = 'title'):

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
                ax.set_title(f'Trial {trial}')
                ax.set_xlabel('Steps')
                ax.set_ylabel('Cumulatve Rewards')
                if legend:
                    ax.legend()
        else:
            ax.axis('off')
        
        i += 1
        trial += 1

    plt.title(title)
    plt.tight_layout()
    plt.show()


def eval_plots_all(image_list, all_rewards=False):
    for image in image_list:
        for ep in image:
            steps, rsum, rewards = image[ep]
            all_steps = [i for i in range(steps)]

            if not all_rewards:
                plt.plot(all_steps, rsum)
                plt.ylabel('Cumulative Rewards')
                plt.xlabel('Steps')
                plt.title('Cumulative Rewards per Step')
            else:
                plt.title('Rewards per Step')
                plt.plot(all_steps, rewards)
                plt.ylabel('Rewards')
                plt.xlabel('Steps')
    plt.show()


def save_episode_plots2(ep_recorder: dict, 
                       save_dir: str, 
                       trial: int,
                       legend: bool =True, 
                       get_all_rewards: bool = True,
                       file_extention: str = '.svg'):
    
    for k in ep_recorder.keys():
        steps, rsum, all_rewards = ep_recorder[k]
        all_steps = [i for i in range(steps)]

        if not get_all_rewards:
            plt.plot(all_steps, rsum, label=f'episode: {k}')
            plt.title(f'Trial {trial} Cumulative Rewards per Step: All Episodes')
            plt.ylabel('Cumulative Rewards')
            plt.xlabel('Steps')
            if legend:
                plt.legend()
        else:
            plt.plot(all_steps, all_rewards, label=f'episode: {k}')
            plt.title(f'Trial {trial} Cumulative Rewards per Step: All Episodes')
            plt.ylabel('Cumulative Rewards')
            plt.xlabel('Steps')
            if legend:
                plt.legend()

    save_path = os.path.join(save_dir, f'trial_{trial}file_extention')
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
    all_ep_rewards = []
    all_ep_total_steps = []
    all_ep_reward_sum = 0

    for episode in ep_recorder:
        total_steps, rsums_per_step, _ = ep_recorder[episode]

        # The last value of the running sum = total of episode rewards.
        all_ep_rewards.append(rsums_per_step[-1])
        all_ep_total_steps.append(total_steps)
        all_ep_reward_sum += rsums_per_step[-1]

    all_ep_rewards_mean = np.mean(all_ep_rewards)
    all_ep_std = np.std(all_ep_rewards)
    all_ep_steps_mean = np.mean(all_ep_total_steps)

    if return_values:
        return all_ep_rewards_mean, all_ep_std, all_ep_reward_sum, all_ep_steps_mean
    else:
        return {'Mean episode rewards': all_ep_rewards_mean,
                'Std episode rewards': all_ep_std,
                'Total trial rewards': all_ep_reward_sum,
                'Mean episode steps': all_ep_steps_mean}



def add_to_recorder(recorder:dict, param_dict:dict):
    for k in param_dict.keys():
        if k not in recorder:
            recorder[k] = []
            recorder[k].append(param_dict[k])
        else:
            recorder[k].append(param_dict[k])


def get_var_name(obj, namespace):
    return [name for name, value in namespace.items() if value is obj][0]


def calculate_grid_size(num_items, num_columns = 4):
    if num_items <= num_columns:
        num_columns = num_items
    num_rows = int(np.ceil(num_items / num_columns))
    return num_rows, num_columns
