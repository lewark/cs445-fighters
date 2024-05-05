from numpy.random import default_rng as rng
from itertools import product
import torch
import pickle
import gzip
import gc

from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm
    

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

    def __init__(self, model_class: BaseAlgorithm, base_env: Env, try_gpu = True, verbose = False):
        self.model_class = model_class
        self.base_env = base_env
        self.env = None
        self.eval_env = None

        self.model = None
        self.model_ops = {'verbose': 0}

        self.train_opts = {'progress_bar': False}

        self._env_func = None
        self.env_func_options = {'render_mode': None}

        self.eval_func = None
        self.eval_func_options = {'deterministic':False}
        self.device = None

        if try_gpu:
            try:
                self.device = get_device()
                if verbose:
                    print(f'running experiments on {self.device}')
            except:
                if verbose:
                    print('unable to set device.')
        
    def set_model_ops(self, model_ops: dict):
        self.model_ops = model_ops
    
    def set_env_func_options(self, options: dict):
        self.set_env_func_options = options

    def set_env_func(self, env_func, options):
        self.set_env_func = env_func
        self.env_func_options = options
    
    def set_eval_func_options(self, options: dict):
        self.eval_func_options = options
    
    def build_model(self, env, hyper_params):
        self.model = self.model_class(env=env, device=self.device, **self.model_ops, **hyper_params)
    
    def get_model(self):
        return self.model

    def save_model(self, save_dir: str):
        if self.model is None:
            print('No model set to save!')
        else:
            self.model.save(save_dir)
            print(f'Model Saved to: {save_dir}')
        
    def train_model(self, total_timesteps: int, hyper_params: dict):

        try:
            print('Learning...')
            self.model_params = hyper_params
            self.env = self.make_env_func(**self.env_func_options)
            self.build_model(env=self.env, hyper_params=hyper_params)
            self.model.learn(total_timesteps=total_timesteps, **self.train_opts)
            
            self.env.close()

        except:
            print('unable to train model, closing env')
            if self.env is not None:
                self.env.close()
        
    def evaluate_model(self, n_eval_episodes=5):
        try:
            print('Evaluating...')
            self.eval_env = self.make_env_func(**self.env_func_options)
            results = self.eval_func(self.model, self.eval_env, n_eval_episodes=n_eval_episodes, **self.eval_func_options)
            self.eval_env.close()

            return results
        except:
            print('unable to evaluate model, closing eval_env')
            if self.eval_env is not None:
                self.eval_env.close()

