import sys

from stable_baselines3 import A2C
from ..common.fighter_envs import StreetFighter, make_env
from gymnasium.wrappers import RecordVideo

VIDEO_DIR = "results/video"

model = A2C.load(sys.argv[1])

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

def make_rec_env():
    return StreetFighter(render_mode="rgb_array", video_folder=VIDEO_DIR)

try:
    vec_env =  make_env(make_rec_env, n_procs=0)
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)
        if dones[0]:
            break
finally:
    vec_env.close()
