import os
import time
from typing import Any, Optional, Union

import cv2
from gymnasium import Env
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat
from gymnasium.spaces import MultiBinary, Box
import numpy as np


class FighterEnv(Env):
    def __init__(self, game: str, render_mode: Optional[str] = "human") -> None:
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)

        import retro
        self.game = retro.make(game=game,
                               use_restricted_actions=retro.Actions.FILTERED,
                               render_mode=render_mode)

    def reset(self, seed: Optional[int] = None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.game.reset(seed=seed)

        obs = self.preprocess(obs)
        self.previous_frame = obs

        self.score = 0

        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.game.step(action)

        obs = self.preprocess(obs)

        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        # plain score-based reward already implemented in default stable-retro config
        # modify to include health
        score = self.compute_score(info)
        reward = score - self.score
        self.score = score

        return frame_delta, reward, terminated, truncated, info

    def render(self, *args, **kwargs) -> Union[RenderFrame, list[RenderFrame], None]:
        return self.game.render(*args, **kwargs)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels

    def seed(self, seed: int) -> None:
        pass

    def close(self) -> None:
        self.game.close()

    def compute_score(self, info: dict[str, Any]) -> int:
        return (
            info.get("score", 0)
            + (info.get("health", 0) - info.get("enemy_health", 0)) * 10
            + (info.get("rounds_won", 0) - info.get("enemy_rounds_won", 0)) * 100
            + (info.get("matches_won", 0) - info.get("enemy_matches_won", 0)) * 1000
        )


class StreetFighter(FighterEnv):
    def __init__(self, render_mode: Optional[str] = "human") -> None:
        super().__init__('StreetFighterIISpecialChampionEdition-Genesis', render_mode)


class ArtOfFighting(FighterEnv):
    def __init__(self, render_mode: Optional[str] = "human") -> None:
        super().__init__('ArtOfFighting-Snes', render_mode)


class MortalKombat3(FighterEnv):
    def __init__(self, render_mode: Optional[str] = "human") -> None:
        super().__init__('MortalKombat3-Genesis', render_mode)


class VirtuaFighter(FighterEnv):
    def __init__(self, render_mode: Optional[str] = "human") -> None:
        super().__init__('VirtuaFighter-32x', render_mode)


def run(env: Env) -> None:
    obs, info = env.reset()
    terminated = False

    for game in range(1):
        while not terminated:
            if terminated:
                obs, info = env.reset()
            env.render()
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            #print(info)
            if reward != 0:
                print(reward)


def make_env(env_class, n_procs: int = 4, n_stack: int = 4, **kwargs) -> Env:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
    
    if n_procs == 0:
        env = DummyVecEnv([lambda: Monitor(env_class(**kwargs), LOG_DIR)])
    else:
        env = SubprocVecEnv([(lambda: Monitor(env_class(**kwargs), LOG_DIR)) for proc in range(n_procs)])
    env = VecFrameStack(env, n_stack, channels_order='last')
    
    return env
