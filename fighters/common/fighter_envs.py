from typing import Any, Optional, Union
import random

import cv2
from gymnasium import Env
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat
from gymnasium.spaces import MultiBinary, Box
import numpy as np
import retro
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
import torch

from .constants import LOG_DIR


class FighterEnv(Env):
    def __init__(self, game: str, render_mode: Optional[str] = "human", random_delay: int = 30, use_delta=False, info=None) -> None:
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.wins = 0
        self.random_delay = random_delay
        self.use_delta = use_delta
        self.game = retro.make(game=game,
                               use_restricted_actions=retro.Actions.FILTERED,
                               render_mode=render_mode,
                               info=info)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs, info = self.game.reset(seed=seed)
        obs = self.preprocess(obs)
        self.previous_frame = obs

        if self.use_delta:
            obs = self.compute_delta(obs)

        self.score = 0

        if self.random_delay > 0:
            null_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)

            for i in range(random.randint(0, self.random_delay)):
                obs, _, _, _, info = self.step(null_action)

        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.game.step(action)

        obs = self.preprocess(obs)

        if self.use_delta:
            obs = self.compute_delta(obs)

        # plain score-based reward already implemented in default stable-retro config
        # modify to include health
        reward = self.compute_reward(info)
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs) -> Union[RenderFrame, list[RenderFrame], None]:
        return self.game.render(*args, **kwargs)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels

    def compute_delta(self, img: np.ndarray) -> np.ndarray:
        delta = img - self.previous_frame
        self.previous_frame = img
        return delta

    def seed(self, seed: int) -> None:
        pass

    def close(self) -> None:
        self.game.close()

    def get_wins(self) -> int:
        return self.wins
        
    def compute_reward(self, info: dict[str, Any]) -> int:
        return (
            info.get("score", 0) / 10
            + (info.get("health", 0) - info.get("enemy_health", 0))
            + (info.get("rounds_won", 0) - info.get("enemy_rounds_won", 0)) * 500
            + (info.get("matches_won", 0) - info.get("enemy_matches_won", 0)) * 1000
        )


class StreetFighter(FighterEnv):
    def __init__(self, render_mode: Optional[str] = "human", random_delay: int = 30, use_delta: bool = False) -> None:
        super().__init__('StreetFighterIISpecialChampionEdition-Genesis', render_mode, random_delay,
                         info="integrations/StreetFighterII.json")
        self.score = 0
        self.enemy_health = 175
        self.health = 175
        self.enemy_wins = 0
        self.player_wins = 0
        self.distance = 307 - 205
        self.random_delay = random_delay

    def compute_reward(self, info: dict[str, Any]) -> int:
        distance = self.get_player_distance(info)
        reward = (distance - self.distance) / 10
        self.distance = distance

        new_health = info["health"]
        if new_health < self.health and new_health != 0:
            reward -= 1
        self.health = new_health

        new_enemy_health = info["enemy_health"]
        if new_enemy_health < self.enemy_health and new_enemy_health != 0:
            reward += 1
        self.enemy_health = new_enemy_health

        new_player_wins = info["matches_won"]
        if new_player_wins > self.player_wins:
            reward += 1
        self.player_wins = new_player_wins

        new_enemy_wins = info["enemy_matches_won"]
        if new_enemy_wins > self.enemy_wins:
            reward -= 1
        self.enemy_wins = new_enemy_wins

        return reward

    def get_player_distance(self, info):
        return np.hypot(
            info["enemy_x"] - info["player_x"],
            info["enemy_y"] - info["player_y"]
        )
        #return abs(info["enemy_x"] - info["player_x"])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        self.score = info.get("score", 0)
        self.enemy_health = info.get("enemy_health", 176)
        self.health = info.get("health", 176)
        self.enemy_wins = info.get("enemy_matches_won", 0)
        self.player_wins = info.get("matches_won", 0)
        self.distance = 307 - 205

        return obs, info


class ArtOfFighting(FighterEnv):
    def __init__(self, render_mode: Optional[str] = "human", random_delay: int = 30, use_delta: bool = False, info=None) -> None:
        super().__init__('ArtOfFighting-Snes', render_mode, random_delay, use_delta, info)


class MortalKombat3(FighterEnv):
    def __init__(self, render_mode: Optional[str] = "human", random_delay: int = 30, use_delta: bool = False, info=None) -> None:
        super().__init__('MortalKombat3-Genesis', render_mode, random_delay, use_delta, info)


class VirtuaFighter(FighterEnv):
    def __init__(self, render_mode: Optional[str] = "human", random_delay: int = 30, use_delta: bool = False, info=None) -> None:
        super().__init__('VirtuaFighter-32x', render_mode, random_delay, use_delta, info)
        

def run(env: Env) -> None:
    obs, info = env.reset()
    terminated = False

    for game in range(1):
        while not terminated:
            if terminated:
                obs, info = env.reset()
            env.render()
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            cv2.imshow('obs', obs)
            cv2.waitKey(1)

            if reward != 0:
                #print(info)
                print(reward)


def make_env(env_class, n_procs: int = 4, n_stack: int = 4, **kwargs) -> Env:
    if n_procs == 0:
        env = DummyVecEnv([lambda: Monitor(env_class(**kwargs), LOG_DIR)])
    else:
        env = SubprocVecEnv([(lambda: Monitor(env_class(**kwargs), LOG_DIR)) for proc in range(n_procs)])
    env = VecFrameStack(env, n_stack, channels_order='last')
    return env
