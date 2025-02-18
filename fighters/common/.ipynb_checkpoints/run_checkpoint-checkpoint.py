import sys

from stable_baselines3 import A2C, DQN, PPO
from ..common.fighter_envs import StreetFighter, make_env
from gymnasium.wrappers import RecordVideo

MODELS = {
    "a2c": A2C,
    "dqn": DQN,
    "ppo": PPO
}

model_class = MODELS[sys.argv[1]]

model_filename = sys.argv[2]
print("Loading model from", model_filename)

model = model_class.load(model_filename)

video_dir = None
if len(sys.argv) >= 4:
    video_dir = sys.argv[3]
    print("Writing video files to", video_dir)

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

def make_fighter_env():
    if video_dir is None:
        return StreetFighter(render_mode="human")
    return StreetFighter(render_mode="rgb_array", video_folder=video_dir)

try:
    vec_env = make_env(make_fighter_env, n_procs=0)
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)
        #if video_dir is not None:
        #    vec_env.render()
        if dones[0]:
            break
finally:
    vec_env.close()
