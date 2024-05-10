import copy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_parameter_grid(data):
    results = copy.deepcopy(data)
    non_params = ['Policy', 'Seed', 'Episode Rewards', 'Episode Stds',
                  'Mean Reward', 'Std Mean', 'Total Reward', 'Elapsed Time']
    columns = results.columns.difference(non_params)
    num_cols = len(columns)
    num_plots = num_cols * (num_cols - 1) //2
    for col in columns:
        results[col] = results[col].astype(float)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 6*num_plots), subplot_kw=dict(projection='3d'))

    index = 0
    for i, col_x in enumerate(columns):
        for j, col_y in enumerate(columns):
            if i < j:
                ax = axs[index]
                x = np.array(results[col_x])
                y = np.array(results[col_y])
                means = np.array(results[non_params[1]])
                means = means.astype(float)
                ax.scatter(x, y, means)
                ax.set_xlabel(col_x)
                ax.set_ylabel(col_y)
                ax.set_title(f'Reward Means: {col_x} vs {col_y}')
                index += 1

    plt.tight_layout()
    plt.show()


def plot_reward_mean(results, param_name, policy_type = False):
    param = np.array(results[param_name])
    param = param.astype(str)
    means = np.array(results['Mean Reward'])

    plt.figure(figsize=(6, 6))
    plt.plot(param, means, 'o')
    plt.xlabel(param_name)
    plt.ylabel("Mean Reward")
    if policy_type is False:
        plt.title(f'Reward Means Associated with {param_name}')
    else:
        policy = results['Policy'].iloc[0]
        plt.title(f'{policy} Reward Means Associated with {param_name}')
    plt.tight_layout()


def plot_reward_std(results, param_name, policy_type = False):
    param = np.array(results[param_name])
    param = param.astype(str)
    std = np.array(results['Std Mean'])

    plt.figure(figsize=(6, 6))
    plt.plot(param, std, 'o')
    plt.xlabel(param_name)
    plt.ylabel("Mean Standard Deviation")
    if policy_type is False:
        plt.title(f'Mean Standard Deviations Associated with {param_name}')
    else:
        policy = results['Policy'].iloc[0]
        plt.title(f'{policy} Mean Standard Deviations Associated with {param_name}')
    plt.tight_layout()


def test():
    rows = 10
    columns = 6
    min_value = 0.0
    max_value = 1.0
    matrix = np.random.uniform(min_value, max_value, size=(rows, columns))
    policy = ['MlpPolicy'] * 5
    policy = policy.extends(['CnnPolicy'] * 5)
    policy = np.vstack(policy)
    whole = np.hstack((policy, matrix))
    tester = pd.Dataframe(whole)
    tester.columns = ['Policy', 'Learning Rate', 'Steps', 'Batch Size',
                         'Mean Reward', 'Total Reward', 'Elapsed Time']
    
    plot_reward_mean(tester, 'Learning Rate')

    plot_parameter_grid(tester)