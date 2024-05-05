from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import vec_monitor, vec_normalize, vec_frame_stack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, ClipRewardEnv, StickyActionEnv

from classes.wrappers import DiscreteWrapper


def make_env(base_env_class, **kwargs):
    return base_env_class(**kwargs)


def make_dqn_env(base_env_class, 
                 render_mode = None, 
                 log_dir = None, 
                 n_procs: int = 0, 
                 n_stack: int = 4
                 ):

    if n_procs == 0:
        env = make_env(base_env_class=base_env_class, render_mode=None)
        env = DiscreteWrapper(env)
        env = ClipRewardEnv(env)
        env = MaxAndSkipEnv(env, 4)
        env = Monitor(env, filename=log_dir)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        return env
    else:
        env = SubprocVecEnv([lambda: Monitor(DiscreteWrapper(make_env(base_env_class, render_mode), log_dir)) for proc in range(n_procs)])
        env = VecFrameStack(env, n_stack, channels_order='last')
        env = VecTransposeImage(env)
        return env
    