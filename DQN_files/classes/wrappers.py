import time
import numpy as np
import gymnasium as gym


# Based on example from https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
class DiscreteWrapper(gym.ActionWrapper):
    """
        Wrap a gymnasium environment to allow it to use discrete a action space.
    """

    def __init__(self, env, input_buttons = None, input_combos = None):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        
        self.buttons = None
        self.combos = None
        self.decode_discrete_action = []

        if input_buttons is None or input_combos is None:
            self.buttons, self.combos = self.make_buttons_and_combos()
        
        for combo in self.combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[self.buttons.index(button)] = True
            self.decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(
            len(self.decode_discrete_action))
        
        
    def action(self, act):
        return self.decode_discrete_action[act].copy()
        

    def make_buttons_and_combos(self):
        action_buttons = ['A', 'B', 'C', 'X', 'Y', 'Z']
        movement_buttons = ['U', 'D', 'L', 'R']
        movement_combos = []
        all_combos = []

        all_buttons = action_buttons + movement_buttons

        for btn in movement_buttons:
            movement_combos.append([btn])

        for b1 in movement_buttons[:2]:
            for b2 in movement_buttons[2:]:
                movement_combos.append([b1, b2])

        for btn in movement_combos:
            all_combos.append(btn)

        for btn in action_buttons:
            all_combos.append([btn])

        for mv_combo in movement_combos:
            for ac_btn in action_buttons:
                all_combos.append(mv_combo + [ac_btn])

        return all_buttons, all_combos