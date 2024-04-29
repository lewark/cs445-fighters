from itertools import combinations

import gymnasium as gym
import numpy as np


def make_buttons_and_combos(combo_level='low'):

    if combo_level not in ['buttons_only', 'low', 'med', 'high']:
        raise ValueError(
            "combo_level must be set to one of the following: 'buttons_only', 'low', 'med, 'high")

    action_buttons = ['A', 'B', 'C', 'X', 'Y', 'Z']
    movement_buttons = ['U', 'D', 'L', 'R']

    all_buttons = action_buttons + movement_buttons

    if combo_level == 'buttons_only':
        combos = []
        for btn in all_buttons:
            combos.appemd([btn])
        
        return all_buttons, combos
    
    else:

        movement_combos = []

        for btn in movement_buttons:
            movement_combos.append([btn])

        for b1 in movement_buttons[:2]:
            for b2 in movement_buttons[2:]:
                movement_combos.append([b1, b2])

        action_combos = {}
        action_combos[1] = []

        for btn in action_buttons:
            action_combos[1].append([btn])

        if combo_level == 'med':
            for i in range(2, 3):
                action_combos[i] = list(combinations(action_buttons, i))

        if combo_level == 'high':
            for i in range(2, 7):
                action_combos[i] = list(combinations(action_buttons, i))

        ac_combos = []

        for key in action_combos:
            for tups in action_combos[key]:
                new_list = []
                for btn in tups:
                    new_list.append(btn)
                ac_combos.append(new_list)

        all_combos = []

        for btn in movement_buttons:
            all_combos.append([btn])

        for btn in action_buttons:
            all_combos.append([btn])

        for mv_combo in movement_combos:
            for ac_comb in ac_combos:
                all_combos.append(mv_combo + ac_comb)

        return all_buttons, all_combos


def _mock_wrapper(buttons, combos):
    decode_discrete_action = []

    for combo in combos:
        arr = np.array([False] * 12)
        for button in combo:
            arr[buttons.index(button)] = True

        decode_discrete_action.append(arr)

    return gym.spaces.Discrete(len(decode_discrete_action))


# https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
class DiscreteWrapper(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combo_level: increases the number of moves/buttons an agent can press at the same time.
        Waring: Evey level up increases the action space size considerably. 

        options str: combo_level = 'buttons_only', 'low', 'med', 'high'
    """

    def __init__(self, env, combo_level='low', buttons=None, combos=None):
        super().__init__(env)

        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        if buttons is None or combos is None:
            buttons, combos = make_buttons_and_combos(combo_level=combo_level)
        
        self.buttons = buttons
        self.combos = combos
        self.decode_discrete_action = []
        
        for combo in self.combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[self.buttons.index(button)] = True
            self.decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(
            len(self.decode_discrete_action))

    def action(self, act):
        return self.decode_discrete_action[act].copy()
