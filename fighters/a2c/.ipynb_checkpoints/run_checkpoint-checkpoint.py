import sys

from stable_baselines3 import A2C

model = A2C.load(sys.argv[1])
