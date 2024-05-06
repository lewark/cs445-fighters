import sys

from stable_baselines3 import A2C
from ..common.fighter_envs import StreetFighter, make_env

model = A2C.load(sys.argv[1])

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
vec_env = make_env(StreetFighter, n_procs=0, render_mode="human")
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    #vec_env.render("human")
