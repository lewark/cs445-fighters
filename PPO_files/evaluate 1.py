from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# ** From https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb
def _evaluate_original(
    model: BaseAlgorithm,
    num_episodes: int = 100,
    deterministic: bool = True,
) -> float:
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    # This function will only work for a single environment
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    for _ in tqdm(range(num_episodes)):
        episode_rewards = []
        done = False
        # Note: SB3 VecEnv resets automatically:
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
        # obs = vec_env.reset()


        while not done:
            # _states are only useful when using LSTM policies
            # `deterministic` is to use deterministic actions
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, _info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")

    return mean_episode_reward




# modified version of eval with plotting.
def evaluate(
    
    model: BaseAlgorithm,
    eval_env,
    num_episodes: int = 100,
    deterministic: bool = True,
    plot: bool = False,
    trial: int = None
) -> float:
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    # This function will only work for a single environment
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    obs = eval_env.reset()
    all_episode_rewards = []
    plt.close('all')

    for i in tqdm(range(num_episodes)):
        episode_rewards = []
        done = False
        # Note: SB3 VecEnv resets automatically:
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
        # obs = vec_env.reset()

        episode_steps = []
        episode_running_sums = []
        episode_step_num = 0
        episode_reward_running_sum = 0

        while not done:
            # _states are only useful when using LSTM policies
            # `deterministic` is to use deterministic actions
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, _info = eval_env.step(action)
            episode_rewards.append(reward)

            if plot:
                episode_step_num += 1
                episode_steps.append(episode_step_num)
                episode_reward_running_sum += int(reward)
                episode_running_sums.append(episode_reward_running_sum)

        all_episode_rewards.append(sum(episode_rewards))

        if plot:
            
            plt.plot(episode_steps, episode_running_sums, label = f'episode: {i}')
            plt.title(f'Trial {trial} Cumulative Rewards vs Steps')
            plt.ylabel('Cumulative Rewards')
            plt.xlabel('Steps')
            plt.legend()
            

    all_episode_rewards_total = int(sum(all_episode_rewards))
    std = np.std(all_episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)

    if plot:
        plt.show()
        plt.close('all')

        
    # print(f"Mean reward: {mean_episode_reward:.2f} +-{std:.2f} - Num episodes: {num_episodes}")
    # print(f"Total rewards all episodes: {all_episode_rewards_total}\n")

    return mean_episode_reward, std, all_episode_rewards_total