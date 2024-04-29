import torch
from stable_baselines3.common.evaluation import evaluate_policy


LOG_DIR = './logs/'
OPT_DIR = './opt/'

def train_model(model_class, env: Env, model_options: dict[str, Any], total_timesteps: int = 25000):
    env.reset()

    start_time = time.time()

    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    device = torch.device('cuda:0')

    model = model_class(env=env, tensorboard_log=LOG_DIR, verbose=0, device=device, **model_options)
    print("Learning...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    print("Evaluating...")
    ep_rewards, _ = evaluate_policy(model, env, n_eval_episodes=4, return_episode_rewards = True)
    reward_mean = np.mean(np.array(ep_rewards))
    reward_sum = np.sum(np.array(ep_rewards))

    elapsed_time = time.time() - start_time
    hyper_ps = [str(model_options[key]) for key in model_options] + [str(reward_mean)]
    SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format("_".join(hyper_ps)))
    model.save(SAVE_PATH)

    print(f'finished architecture {hyper_ps} at {elapsed_time/60} minutes.')
    del model, env, ep_rewards, reward_mean, reward_sum, _
    torch.cuda.empty_cache()
    return hyper_ps
