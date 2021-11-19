import numpy as np
import gym
import gym.spaces as spaces
import time
from vi_pi_helpers import policy_iteration, update_policy, play_episodes
import matplotlib.pyplot as plt
import json
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv

def discount_experiment(env,dir):
    gamma_vals = [0.2,0.3,0.5,0.8,0.9]
    metrics = {}
    for gamma_val in gamma_vals:
        metrics[gamma_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[], 'num_wins':[]}
        start = time.time()
        opt_V, opt_Policy, converge_iteration, policies = policy_iteration(env, max_iteration = 1000, discount_factor=gamma_val)
        end = time.time()
        generate_plots(policies,env,f'gamma_{gamma_val}',dir=dir)
        # get the metrics for this policy
        win, total_reward, avg_reward = play_episodes(env,10,opt_Policy)
        metrics[gamma_val]['reward'] = total_reward
        metrics[gamma_val]['iteration'] = converge_iteration
        metrics[gamma_val]['time'] = end - start
        metrics[gamma_val]['avg_reward'] = avg_reward
        metrics[gamma_val]['num_wins'] = win
    with open(f'markov_decision_process/charts/{dir}/gamma_metrics.json','w') as f:
        json.dump(metrics,f,indent=4)

def epsilon_experiment(env,dir):
    epsilons = [1,10,20,30,40]
    metrics = {}
    for epsilon_val in epsilons:
        metrics[epsilon_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[], 'num_wins':[]}
        start = time.time()
        opt_V, opt_Policy, converge_iteration, policies = policy_iteration(env, max_iteration = 1000, epsilon_change=epsilon_val)
        end = time.time()
        generate_plots(policies,env,f'epsilon_{epsilon_val}',dir=dir)
        # get the metrics for this policy
        win, total_reward, avg_reward = play_episodes(env,10,opt_Policy)
        metrics[epsilon_val]['reward'] = total_reward
        metrics[epsilon_val]['iteration'] = converge_iteration
        metrics[epsilon_val]['time'] = end - start
        metrics[epsilon_val]['avg_reward'] = avg_reward
        metrics[epsilon_val]['num_wins'] = win
    with open(f'markov_decision_process/charts/{dir}/epsilon_metrics.json','w') as f:
        json.dump(metrics,f,indent=4)


def get_rewards(policies,env):
    wins = []
    rewards = []
    avg_rewards = []
    for policy in policies:
        win, total_reward, avg_reward = play_episodes(env,10,policy)
        wins.append(win)
        rewards.append(total_reward)
        avg_rewards.append(avg_reward)
    return wins,rewards,avg_rewards

def generate_plots(policies,env,figname,dir):
    wins, rewards, avg_rewards = get_rewards(policies,env)
    _,ax = plt.subplots(1)
    ax.plot(avg_rewards)
    ax.set_title('Average Reward')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Avg Reward')
    plt.savefig(f'markov_decision_process/charts/{dir}/{figname}_avg_reward.png')
    plt.clf()

    _,ax = plt.subplots(1)
    ax.plot(rewards)
    ax.set_title('Reward')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Reward')
    plt.savefig(f'markov_decision_process/charts/{dir}/{figname}_reward.png')
    plt.clf()

    _,ax = plt.subplots(1)
    ax.plot(wins)
    ax.set_title('Wins')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('wins')
    plt.savefig(f'markov_decision_process/charts/{dir}/{figname}_wins.png')
    plt.clf()


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    env.seed(50)

    discount_experiment(env,'frozen_lake_small/pi/gamma')
    epsilon_experiment(env,'frozen_lake_small/pi/epsilon')

    np.random.seed(10)
    env = FrozenLakeEnv(generate_random_map(20))

    discount_experiment(env,'frozen_lake_large/pi/gamma')
    epsilon_experiment(env,'frozen_lake_large/pi/epsilon')
