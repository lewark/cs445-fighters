from gymnasium import Env

from ..common.fighter_envs import StreetFighter
from .a2c_envs import CustomReward, AustinReward

from tqdm import tqdm

import numpy as np


def test_env(env: Env) -> None:

    all_timesteps = []
    all_reward = []
    for game in tqdm(range(25)):
        total_timesteps = 0
        total_reward = 0

        obs, info = env.reset()
        terminated = False
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            total_timesteps += 1
            total_reward += reward

        all_timesteps.append(total_timesteps)
        all_reward.append(total_reward)

    mean_reward = np.mean(all_reward)
    std_reward = np.std(all_reward)

    mean_timesteps = np.mean(all_timesteps)
    std_timesteps = np.std(all_timesteps)

    print("Mean reward:", mean_reward, "stdev:", std_reward)
    print("Mean timesteps:", mean_timesteps, "stdev:", std_timesteps)

def mean(items):
    return sum(items) / len(items)


if __name__ == "__main__":
    game = StreetFighter(random_delay=0, render_mode=None)

    env = CustomReward(game, use_distance=False)
    print("Damage and wins")
    test_env(env)

    env = CustomReward(game, use_distance=True)
    print("Damage, wins, and distance")
    test_env(env)

    env = AustinReward(game)
    print("All rewards")
    test_env(env)

    game.close()
