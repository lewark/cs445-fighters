from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.vec_env import VecNormalize
# from stable_baselines3.common.vec_env import vec_monitor, vec_normalize, vec_frame_stack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, ClipRewardEnv

from classes.wrappers import DiscreteWrapper
from classes.constants import LOG_DIR


def make_discrete_env(base_env_class, **kwargs):
    env = base_env_class(**kwargs)
    env = DiscreteWrapper(env)
    return env


def make_dqn_env(base_env_class, 
                 render_mode = None, 
                 log_dir = LOG_DIR, 
                 n_procs: int = 4, 
                 n_stack: int = 4
                 ):

    if n_procs == 0:
        env = make_discrete_env(base_env_class=base_env_class, render_mode=render_mode)
        env = Monitor(env, filename=log_dir)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack, channels_order='last')
        return env
    else:
        env = SubprocVecEnv([lambda: Monitor(make_discrete_env(base_env_class=base_env_class, render_mode=render_mode), log_dir) for proc in range(n_procs)])
        env = VecFrameStack(env, n_stack, channels_order='last')
        # env = VecTransposeImage(env)
        return env


def skip_env(base_env_class, 
                render_mode = None, 
                log_dir = LOG_DIR, 
                n_procs: int = 4, 
                n_stack: int = 4
                ):

    if n_procs == 0:
        env = make_discrete_env(base_env_class=base_env_class, render_mode=render_mode)
        env = MaxAndSkipEnv(env, skip=4)
        env = Monitor(env, filename=log_dir)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        return env
    else:
        env = SubprocVecEnv([lambda: Monitor(MaxAndSkipEnv(make_discrete_env(base_env_class=base_env_class, render_mode=render_mode), 4), log_dir) for proc in range(n_procs)])
        env = VecFrameStack(env, n_stack, channels_order='last')
        env = VecTransposeImage(env)
        return env


def clip_env(base_env_class, 
                render_mode = None, 
                log_dir = LOG_DIR, 
                n_procs: int = 4, 
                n_stack: int = 4
                ):

    if n_procs == 0:
        env = make_discrete_env(base_env_class=base_env_class, render_mode=render_mode)
        env = ClipRewardEnv(env)
        env = Monitor(env, filename=log_dir)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        return env
    else:
        env = SubprocVecEnv([lambda: Monitor(ClipRewardEnv(make_discrete_env(base_env_class=base_env_class, render_mode=render_mode)), log_dir) for proc in range(n_procs)])
        env = VecFrameStack(env, n_stack, channels_order='last')
        env = VecTransposeImage(env)
        return env


def maxskip_and_clip_env(base_env_class, 
                render_mode = None, 
                log_dir = LOG_DIR, 
                n_procs: int = 4, 
                n_stack: int = 4
                ):

    if n_procs == 0:
        env = make_discrete_env(base_env_class=base_env_class, render_mode=None)
        env = ClipRewardEnv(env)
        env = MaxAndSkipEnv(env, 4)
        env = Monitor(env, filename=log_dir)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        return env
    else:
        env = SubprocVecEnv([lambda: 
                             Monitor(MaxAndSkipEnv(ClipRewardEnv(make_discrete_env(base_env_class=base_env_class, render_mode=render_mode)), 4), log_dir) 
                             for proc in range(n_procs)])
        env = VecFrameStack(env, n_stack, channels_order='last')
        env = VecTransposeImage(env)
        return env


class AllDqnEnvFunctions:
    def __init__(self) -> None:
        self._env_func_list = [skip_env, make_dqn_env, maxskip_and_clip_env]

    def get_env_func_list(self):
        return self._env_func_list
    
    def __str__(self) -> str:
        self.str_output = str([str(env_func.__name__) for env_func in self._env_func_list])
        return self.str_output
    
    def __repr__(self) -> str:
        self.str_output = str([str(env_func.__name__) for env_func in self._env_func_list])
        return self.str_output

class EXP_2_all_env_functions:
    def __init__(self) -> None:
        self._env_func_list = [make_dqn_env, skip_env]

    def get_env_func_list(self):
        return self._env_func_list
    
    def __str__(self) -> str:
        self.str_output = str([str(env_func.__name__) for env_func in self._env_func_list])
        return self.str_output
    
    def __repr__(self) -> str:
        self.str_output = str([str(env_func.__name__) for env_func in self._env_func_list])
        return self.str_output


class EXP_3_all_env_functions:
    def __init__(self) -> None:
        self._env_func_list = [make_dqn_env]

    def get_env_func_list(self):
        return self._env_func_list
    
    def __str__(self) -> str:
        self.str_output = str([str(env_func.__name__) for env_func in self._env_func_list])
        return self.str_output
    
    def __repr__(self) -> str:
        self.str_output = str([str(env_func.__name__) for env_func in self._env_func_list])
        return self.str_output


class EXP_4_all_env_functions:
    def __init__(self) -> None:
        self._env_func_list = [make_dqn_env]

    def get_env_func_list(self):
        return self._env_func_list
    
    def __str__(self) -> str:
        self.str_output = str([str(env_func.__name__) for env_func in self._env_func_list])
        return self.str_output
    
    def __repr__(self) -> str:
        self.str_output = str([str(env_func.__name__) for env_func in self._env_func_list])
        return self.str_output