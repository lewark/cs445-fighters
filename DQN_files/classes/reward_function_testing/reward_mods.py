from classes.fighter_envs import StreetFighter, FighterEnv
from typing import Any, Optional, Union
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat
import numpy as np


class SF_FIRST_REW_FUNC(FighterEnv):
    def __init__(self, render_mode: Optional[str] = "human", random_delay: int = 30, use_delta: bool = False) -> None:
        super().__init__('StreetFighterIISpecialChampionEdition-Genesis', render_mode, random_delay,
                         info="integrations/StreetFighterII.json")
        
        self.custom_test_name = 'SF_FIRST_REW_FUNC'

    def __str__(self):
        return self.custom_test_name
    
    def __repr__(self):
        return self.custom_test_name

class SF_FIRST_REW_FUNC_FRAME_DELTA(SF_FIRST_REW_FUNC):
    def __init__(self, render_mode: Optional[str] = None, random_delay: int = 30, use_delta: bool = True):
        super().__init__(render_mode=render_mode, random_delay=random_delay, use_delta=use_delta)
        self.custom_test_name = 'SF_FIRST_REW_FUNC_FRAME_DELTA'

    
class SF_Default(StreetFighter):
    def __init__(self, render_mode: Optional[str] = None, random_delay: int = 30, use_delta: bool = False):
        super().__init__(render_mode=render_mode, random_delay=random_delay, use_delta=use_delta)
        self.custom_test_name = 'Default_No_Frame_Delta'

        self.score = 0
        self.enemy_health = 175
        self.health = 175
        self.enemy_wins = 0
        self.player_wins = 0
        self.random_delay = random_delay

    def __str__(self):
        return self.custom_test_name
    
    def __repr__(self):
        return self.custom_test_name
    
    def compute_reward(self, info: dict[str, Any]) -> int:
        reward = 0

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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        self.score = info.get("score", 0)
        self.enemy_health = info.get("enemy_health", 176)
        self.health = info.get("health", 176)
        self.enemy_wins = info.get("enemy_matches_won", 0)
        self.player_wins = info.get("matches_won", 0)
        self.distance = 307 - 205

        return obs, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        self.score = info.get("score", 0)
        self.enemy_health = info.get("enemy_health", 176)
        self.health = info.get("health", 176)
        self.enemy_wins = info.get("enemy_matches_won", 0)
        self.player_wins = info.get("matches_won", 0)

        return obs, info


class SF_FrameDelta(SF_Default):
    def __init__(self, render_mode: Optional[str] = None, random_delay: int = 30, use_delta: bool = True):
        super().__init__(render_mode=render_mode, random_delay=random_delay, use_delta=use_delta)
        self.custom_test_name = 'Default_With_Frame_Delta'

    def __str__(self):
        return self.custom_test_name
    
    def __repr__(self):
        return self.custom_test_name
    

class SF_Default_Scaled_Rewards(StreetFighter):
    def __init__(self, render_mode: Optional[str] = None, random_delay: int = 30, use_delta: bool = False):
        super().__init__(render_mode=render_mode, random_delay=random_delay, use_delta=use_delta)
        self.custom_test_name = 'SF_Default_Scaled_Rewards'

        self.score = 0
        self.enemy_health = 175
        self.health = 175
        self.enemy_wins = 0
        self.player_wins = 0
        self.random_delay = random_delay

    def __str__(self):
        return self.custom_test_name
    
    def __repr__(self):
        return self.custom_test_name
    
    def compute_reward(self, info: dict[str, Any]) -> int:
        reward = 0

        new_health = info["health"]
        if new_health < self.health and new_health != 0:
            reward -= 2
        self.health = new_health

        new_enemy_health = info["enemy_health"]
        if new_enemy_health < self.enemy_health and new_enemy_health != 0:
            reward += 1
        self.enemy_health = new_enemy_health

        new_player_wins = info["matches_won"]
        if new_player_wins > self.player_wins:
            reward += 5
        self.player_wins = new_player_wins

        new_enemy_wins = info["enemy_matches_won"]
        if new_enemy_wins > self.enemy_wins:
            reward -= 5
        self.enemy_wins = new_enemy_wins

        return reward


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        self.score = info.get("score", 0)
        self.enemy_health = info.get("enemy_health", 176)
        self.health = info.get("health", 176)
        self.enemy_wins = info.get("enemy_matches_won", 0)
        self.player_wins = info.get("matches_won", 0)

        return obs, info


class SF_New_Movement_Data(StreetFighter):
    def __init__(self, render_mode: Optional[str] = None, random_delay: int = 30, use_delta: bool = False):
        super().__init__(render_mode=render_mode, random_delay=random_delay, use_delta=use_delta)
        self.custom_test_name = 'Default_No_Frame_Delta_With_Movement_Data'

    def __str__(self):
        return self.custom_test_name
    
    def __repr__(self):
        return self.custom_test_name
    

class SF_New_Movement_Data_Frame_Delta(StreetFighter):
    def __init__(self, render_mode: Optional[str] = None, random_delay: int = 30, use_delta: bool = True):
        super().__init__(render_mode=render_mode, random_delay=random_delay, use_delta=use_delta)
        self.custom_test_name = 'Default_With_Frame_Delta_And_Movement_Data'

    def __str__(self):
        return self.custom_test_name
    
    def __repr__(self):
        return self.custom_test_name


class SF_New_Movement_Data_Scaled_Rewards(StreetFighter):
    def __init__(self, render_mode: Optional[str] = None, random_delay: int = 30, use_delta: bool = False):
        super().__init__(render_mode=render_mode, random_delay=random_delay, use_delta=use_delta)
        self.custom_test_name = 'Default_No_Frame_Delta_With_Movement_Data_Scaled_Rewards'

        self.score = 0
        self.enemy_health = 175
        self.health = 175
        self.enemy_wins = 0
        self.player_wins = 0
        self.distance = 307 - 205
        self.random_delay = random_delay

    def compute_reward(self, info: dict[str, Any]) -> int:
        distance = self.get_player_distance(info)
        reward = (distance - self.distance) / 100
        self.distance = distance

        new_health = info["health"]
        if new_health < self.health and new_health != 0:
            reward -= 2
        self.health = new_health

        new_enemy_health = info["enemy_health"]
        if new_enemy_health < self.enemy_health and new_enemy_health != 0:
            reward += 1
        self.enemy_health = new_enemy_health

        new_player_wins = info["matches_won"]
        if new_player_wins > self.player_wins:
            reward += 5
        self.player_wins = new_player_wins

        new_enemy_wins = info["enemy_matches_won"]
        if new_enemy_wins > self.enemy_wins:
            reward -= 5
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


class SF_Test_Modules:
    def __init__(self) -> None:
        self.mod_list = []
        self.test_name = 'SF_Test_Modules'

    def get_test_modules(self):
        return self.mod_list

    def __str__(self):
        modules = str([mod.__name__ for mod in self.mod_list])
        return f'{self.test_name}: {modules}'
        
        
    def __repr__(self):
        return self.__str__()


class SF_EXP_0_Test_Modules(SF_Test_Modules):
    def __init__(self) -> None:
        super().__init__()
        self.mod_list = [SF_FIRST_REW_FUNC_FRAME_DELTA]
        self.test_name = 'EXP 0: Test of first reward function w/ frame delta'
                
    
class SF_EXP_1_Test_Modules(SF_Test_Modules):
    def __init__(self) -> None:
        super().__init__()
        self.mod_list = [SF_Default, SF_FrameDelta]
        self.test_name = 'EXP 1: Reward function modules'


class SF_EXP_2_Test_Modules(SF_Test_Modules):
    def __init__(self) -> None:
        super().__init__()
        self.mod_list = [SF_FIRST_REW_FUNC, SF_FIRST_REW_FUNC_FRAME_DELTA]
        self.test_name = 'EXP 2: Reward function modules'


class SF_EXP_3_Test_Modules(SF_Test_Modules):
    def __init__(self) -> None:
        super().__init__()
        self.mod_list = [SF_New_Movement_Data, SF_New_Movement_Data_Frame_Delta]
        self.test_name = 'EXP 3: Reward function modules with movement data'


class SF_EXP_4_Test_Modules(SF_Test_Modules):
    def __init__(self) -> None:
        super().__init__()
        self.mod_list = [SF_Default, SF_New_Movement_Data, SF_Default_Scaled_Rewards, SF_New_Movement_Data_Scaled_Rewards]
        self.test_name = 'EXP 4: Reward function modules with movement data and scaled rewards'