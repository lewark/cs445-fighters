
import torch
import numpy as np
import pandas as pd
from itertools import product
from typing import Any, Optional, Union

import cv2
import retro
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

import fighter_envs

print(torch.cuda.device_count())

def create_ppo_first_options(n_policy, n_learning_rate, batch_size, n_batches, n_epochs):
    models = []
    for policy in n_policy:
        for learning_rate in n_learning_rate:
            for batches in n_batches:
                for epochs in n_epochs:
                    models.append({"policy": policy, "learning_rate": learning_rate, "n_steps": batch_size * batches,
                                   "batch_size": batch_size, "n_epochs": epochs})
    return models

n_policy = ['CnnPolicy', 'MlpPolicy']
n_learning_rate = [0.01, 0.001, 0.0001, 0.00001]
batch_size = [32, 64, 128, 256, 512]
n_batches = [4, 16, 32, 64]
n_epochs = [50, 100, 250, 500]

n_procs = 0
archs = []

models = create_ppo_first_options(n_policy, n_learning_rate, batch_size, n_batches, n_epochs)
count = 1
total = models.shape[0]
for model_options in models:
    print(f'Starting architecture {count} of {total}')
    archs.append(fighter_envs.train_model(PPO, fighter_envs.StreetFighter, model_options, n_procs))
    count+=1
    

print(archs)

# def create_ppo_sec_options(n_gamma, n_gae_lambda, n_clip_range):
#     models = []
#     for gamma in n_gamma:
#         for gae_lambda in n_gae_lambda:
#             for clip_range in n_clip_range:
#                     models.append({"policy": policy, "learning_rate": learning_rate, "n_steps": batch_size * batches,
#                                    "batch_size": batch_size, "n_epochs": epochs})
#     return models

# n_gamma = [0.3, 0.8, 0.33, 0.88, 0.333, 0.888, 0.3333, 0.8888]
# n_gae_lambda = [0.8, 0.85, 0.9, 0.95, 0.999]
# n_clip_range = [0.1, 0.2, 0.3, 0.4]