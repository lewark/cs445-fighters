from typing import Any, Optional

import numpy as np
from gymnasium.core import ActType, ObsType

from ..common.fighter_envs import StreetFighter



class CustomReward(Wrapper):
    def __init__(self, env, reward_function) -> None:
        super().__init__(env)
        self.reward_function = reward_function

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.reward_function.reset(info)
        return obs, info

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward_function.compute_reward(info)
        return obs, reward, terminated, truncated, info


class RewardFunction:
    def __init__(self, use_distance: bool) -> None:
        self.player_score = 0
        self.enemy_health = 175
        self.health = 175
        self.enemy_wins = 0
        self.player_wins = 0
        self.player_y = 0
        self.distance = 307 - 205
        self.use_distance = use_distance

    def compute_reward(self, info: dict[str, Any]) -> int:
        print("compute_reward")
        reward = 0

        if self.use_distance:
            distance = self.get_player_distance(info)
            reward = (distance - self.distance) / 10
            self.distance = distance

        new_health = info["health"]
        new_enemy_health = info["enemy_health"]
        new_player_wins = info["matches_won"]
        new_enemy_wins = info["enemy_matches_won"]

        if new_health < self.health and new_health != 0:
            reward -= 1
        self.health = new_health

        if new_enemy_health < self.enemy_health and new_enemy_health != 0:
            reward += 1
        self.enemy_health = new_enemy_health

        if new_player_wins > self.player_wins:
            reward += 1
        self.player_wins = new_player_wins

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

    def reset(self, info) -> None:
        self.player_score = info.get("score", 0)
        self.enemy_health = info.get("enemy_health", 176)
        self.health = info.get("health", 176)
        self.enemy_wins = info.get("enemy_matches_won", 0)
        self.player_wins = info.get("matches_won", 0)
        self.distance = 307 - 205


class AustinRewardFunction(RewardFunction):
    def __init__(self):
        super().__init__(True)

    def compute_reward(self, info: dict[str, Any]) -> int:
        distance = self.get_player_distance(info)
        reward = (distance - self.distance) / 10
        self.distance = distance

        new_health = info["health"]
        new_enemy_health = info["enemy_health"]
        new_player_wins = info["matches_won"]
        new_enemy_wins = info["enemy_matches_won"]
        new_y = info["player_y"]

        if new_y < 192 and new_health < self.health:
            reward += -0.5
        self.player_y = new_y

        if (self.health - new_health) < 15 and (self.health - new_health) > 0:
            reward += 1

        if new_health == self.health and new_enemy_health == self.enemy_health:
            if new_health > new_enemy_health and (new_health != 0 and new_enemy_health != 0):
                reward += 0.2
            elif new_health < new_enemy_health and (new_health != 0 and new_enemy_health != 0):
                reward += -0.2

        # if self.steps >= 200000:
        if new_health < self.health and new_health > 0:
            reward += -0.8
        self.health = new_health

        if new_enemy_health < self.enemy_health and new_enemy_health > 0:
            reward += 0.8
        self.enemy_health = new_enemy_health

        if new_player_wins > self.player_wins:
            reward += 2 * new_player_wins
        self.player_wins = new_player_wins

        if new_enemy_wins > self.enemy_wins:
            reward += -2 * new_enemy_wins
        self.enemy_wins = new_enemy_wins

        return reward
