from numpy.random import default_rng as rng
from itertools import product
from typing import Any, Callable, Dict
import torch
import pickle
import gzip
import gc

from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from classes.constants import TB_DIR, OPT_DIR
    

# Build based on https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
def build_all_permutations(param_dict: dict):

    """
    A function to build all possible permutations of a
    param dict, returned as a list of param dicts.

    It is a useful function for building all possible
    combinations of a model's hyperparameters for traditional
    grid search.

    Input: a param dict with each hyperparameter to be searched as
    a key, and each value to be searched as a list ov values assigned
    to that key.

    Output: A list of param dicts for each possible combination of
    hyperparameters provided in the input param dict.
    """
    keys, values = zip(*param_dict.items())
    experiments = list(dict(zip(keys, v)) for v in product(*values))

    return experiments


def get_env_attributes(env):
    env.reset()
    results = env.step(env.action_space.sample().reshape(1,))
    return results[-1]


def add_to_recorder(recorder: dict, param_dict: dict):
    for k in param_dict.keys():
        if k not in recorder:
            recorder[k] = []
            recorder[k].append(param_dict[k])
        else:
            recorder[k].append(param_dict[k])


def get_var_name(obj, namespace):
    return [name for name, value in namespace.items() if value is obj][0]


def get_results_dict(params: tuple, namespace):
    param_dict = {}

    for p in params:
        p_name = get_var_name(p, namespace)
        param_dict[str(p_name)] = p

    return param_dict


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
            device = torch.device('cuda:0')
    return device


def cleanup_device():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# from https://gist.github.com/thearn/5424244
def save(object, filename, protocol = 0):
        """Saves a compressed object to disk
        """
        file = gzip.GzipFile(filename, 'wb')
        file.write(pickle.dumps(object, protocol))
        file.close()


def load(filename):
        """Loads a compressed object from disk
        """
        file = gzip.GzipFile(filename, 'rb')
        buffer = ""
        while True:
                data = file.read()
                if data == "":
                        break
                buffer += data
        object = pickle.loads(buffer)
        file.close()
        return object


def save_results_as_dict(image_list):
    image_recorder = {}

    for trial, image in enumerate(image_list):
        for episode in image:
            steps, rsum, all_rewards, _ = image[episode]

            all_steps = [i for i in range(steps)]
            image_recorder[(trial, episode, 0)] = all_steps
            image_recorder[(trial, episode, 1)] = rsum
            image_recorder[(trial, episode, 2)] = all_rewards
    
    return image_recorder


def save_trial_data(image_list):
    image_recorder = {}

    for trial, image in enumerate(image_list):
        for episode in image:

            results = image[episode]
            for i, elem in enumerate(results):
                image_recorder[(trial, episode, i)] = elem
    
    return image_recorder


def kwargs_to_dict(**kwargs):
        return kwargs



class RandomGridSearch:
    """
        Build randomly selected options from a param dict.

        Input: param_dict.

        Return: A list of randomized param dicts with
        RandomGridSearch.get_random_params()
    """

    def __init__(self, params_dict: dict, seed: int) -> None:
        self.params_dict = params_dict.copy()
        self.rng = rng(seed)


    def _build_params(self, values: list):
        params = {}

        index = 0
        for key in self.params_dict.keys():
            params[key] = values[index]
            index += 1

        return params

    def _build_random_parma_list(self):
        param_list = []

        for key in self.params_dict.keys():
            param = self.rng.choice(
                self.params_dict[key], size=1, replace=False)
            param = param.tolist()
            param_list.append(param[0])

        return param_list
    
    def get_random_params(self, num_param_sets: int):
        """
        param: num_param_sets: int. The number of randomized
        parm dicts to generate from the input param dict.

        Returns: A list of randomized param dictionary objects
        where one item from each key in the param dict provided
        to the constructor is added to each new param dict.
        
        """
        param_set_list = []
        for i in range(num_param_sets):
            parm_list = self._build_random_parma_list()
            param_set_list.append(self._build_params(parm_list))
        
        return param_set_list



class ExperimentRunner:

    def __init__(self, model_class: BaseAlgorithm=None, base_env: Env =None , try_gpu = True, verbose = False, results_dir = './results/'):
        self.verbose = verbose
        self.model_class = model_class
        self.base_env = base_env
        self.env = None
        self.eval_env = None

        self.results_dir = results_dir
        self.tb_log_name = "DQN"

        self.model = None
        self.model_params = None
        self.model_ops = {'tensorboard_log': TB_DIR, 'verbose': 0}
        
        self.train_opts = {'total_timesteps': 50_000,'progress_bar': True, 'tb_log_name':self.tb_log_name}

        self.env_func = None
        self.env_func_options = {'render_mode': None}

        self.eval_func = evaluate_policy
        self.eval_func_options = {'deterministic':False}
        self.device = None

        self.learn_timesteps = 0

        if try_gpu:
            try:
                self.device = get_device()
                if verbose:
                    print(f'running experiments on {self.device}')
            except:
                if verbose:
                    print('unable to set device.')
    
    def set_model_class(self, model_class: BaseAlgorithm):
        self.model_class = model_class
    
    def set_base_env(self, base_env: Env):
        self.base_env = base_env

    def set_model_ops(self, tensorboard_log=TB_DIR, verbose = 0, **kwargs):
        model_ops = kwargs_to_dict(tensorboard_log=tensorboard_log, verbose = verbose, **kwargs)
        self.model_ops = model_ops
    
    def set_env_func_options(self, render_mode:str = None, **kwargs):
        opts = kwargs_to_dict(render_mode=render_mode, **kwargs)
        self.env_func_options = opts

    def set_env_func(self, env_func: Callable):
        self.env_func = env_func

    def set_eval_func_options(self, return_episode_rewards=False, deterministic=False, **kwargs):
        opts = kwargs_to_dict(return_episode_rewards=return_episode_rewards, deterministic=deterministic, **kwargs)
        self.eval_func_options = opts
    
    def set_eval_func(self, eval_func = evaluate_policy):
        self.eval_func = eval_func
    
    def build_model(self, env, hyper_params):
        self.model = self.model_class(env=env, device=self.device, **self.model_ops, **hyper_params)
    
    def set_tb_log_name(self, tb_log_name: str):
        self.tb_log_name = tb_log_name
        self.train_opts['tb_log_name'] = tb_log_name

    def set_tran_opts(self, total_timesteps: int = 500_000, progress_bar=True, tb_log_name = 'DQN', **kwargs):
        opts = kwargs_to_dict(total_timesteps=total_timesteps, progress_bar=progress_bar, tb_log_name = tb_log_name, **kwargs)
        self.train_opts = opts
    
    def get_model(self):
        return self.model

    def save_model(self, save_dir: str = OPT_DIR):
        if self.model is None:
            print('No model set to save!')
        else:
            self.model.save(save_dir)
            if self.verbose:
                print(f'Model Saved to: {save_dir}')
        
    def train_model(self, hyper_params: dict, total_timesteps: int = None,  keep_model:bool = False):

        if total_timesteps is not None:
            self.train_opts['total_timesteps'] = total_timesteps

        try:
            print('Learning...')
            if keep_model:
                self.learn_timesteps += total_timesteps
                if self.env is None:
                    self.env = self.env_func(self.base_env, **self.env_func_options)
                else:
                    self.env.reset()

                if self.model is None:
                    self.build_model(env=self.env, hyper_params=hyper_params)

                self.model.learn(reset_num_timesteps=False, **self.train_opts)
            else:
                self.model_params = hyper_params
                if self.env is None:
                    self.env = self.env_func(self.base_env, **self.env_func_options)

                self.build_model(env=self.env, hyper_params=hyper_params)
                self.model.learn(**self.train_opts)
                self.learn_timesteps = total_timesteps
            
                self.env.close()
                self.env = None

        except Exception as e:
            print('unable to train model, closing env')
            if self.env is not None:
                self.env.close()
            print(e.with_traceback())
        
    def evaluate_model(self, n_eval_episodes=25):
        try:
            print('Evaluating...')
            self.eval_env = self.env_func(self.base_env, **self.env_func_options)
            results = self.eval_func(self.model, self.eval_env, n_eval_episodes=n_eval_episodes, **self.eval_func_options)
            self.eval_env.close()

            return results
        except Exception as e:
            print('unable to evaluate model, closing eval_env')
            if self.eval_env is not None:
                self.eval_env.close()
            print(e.with_traceback())

    def __str__(self):
        self.str_output = f'expr: model: {self.model_class},\n'
        self.str_output = self.str_output + f'base_env_class: {self.base_env}'
        self.str_output = self.str_output + f'h_parameters: {self.model_params},\n'
        total_time_steps = self.train_opts['total_timesteps']
        self.str_output = self.str_output + f'total_timesteps: {total_time_steps},\n'
        self.str_output = self.str_output + f'model current learning timesteps: {self.learn_timesteps}'

        return self.str_output
    
    def __repr__(self):
        self.str_output = f'expr: model: {self.model_class},\n'
        self.str_output = self.str_output + f'base_env_class: {self.base_env}\n'
        self.str_output = self.str_output + f'h_parameters: {self.model_params},\n'
        self.str_output = self.str_output + f'model_ops: {self.model_ops},\n'
        self.str_output = self.str_output + f'model_train_ops: {self.train_opts},\n'
        if self.env_func is not None:
            self.str_output = self.str_output + f'env_builder: {self.env_func.__name__},\n' 
        self.str_output = self.str_output + f'env_builder opts: {self.env_func_options},\n'
        if self.eval_func is not None:
            self.str_output = self.str_output + f'eval function: {self.eval_func.__name__},\n'
        self.str_output = self.str_output + f'eval function opts: {self.eval_func_options},\n'
        self.str_output = self.str_output + f'device: {self.device},\n'
        self.str_output = self.str_output + f'model current learning timesteps: {self.learn_timesteps}'

        return self.str_output

