import numpy as np
import gym
import gym.spaces as spaces
import time
from vi_pi_helpers import value_iteration, update_policy, play_episodes
import matplotlib.pyplot as plt
import json

def discount_experiment(env,dir):
    gamma_vals = [0.2,0.3,0.5,0.8,0.9]
    metrics = {}
    for gamma_val in gamma_vals:
        metrics[gamma_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[], 'num_wins':[]}
        start = time.time()
        opt_V, opt_Policy, converge_iteration, value_funcs = value_iteration(env, max_iteration = 1000, discount_factor=gamma_val)
        end = time.time()
        generate_plots(value_funcs,env,f'gamma_{gamma_val}',dir=dir)
        # get the metrics for this policy
        win, total_reward, avg_reward = play_episodes(env,10,opt_Policy)
        metrics[gamma_val]['reward'] = total_reward
        metrics[gamma_val]['iteration'] = converge_iteration
        metrics[gamma_val]['time'] = end - start
        metrics[gamma_val]['avg_reward'] = avg_reward
        metrics[gamma_val]['num_wins'] = win
    with open(f'markov_decision_process/charts/frozen_lake/{dir}/gamma_metrics.json','w') as f:
        json.dump(metrics,f,indent=4)

def epsilon_experiment(env,dir):
    epsilons = [1,10,20,30,40]
    metrics = {}
    for epsilon_val in epsilons:
        metrics[epsilon_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[], 'num_wins':[]}
        start = time.time()
        opt_V, opt_Policy, converge_iteration, value_funcs = value_iteration(env, max_iteration = 1000, epsilon_change=epsilon_val)
        end = time.time()
        generate_plots(value_funcs,env,f'epsilon_{epsilon_val}',dir=dir)
        # get the metrics for this policy
        win, total_reward, avg_reward = play_episodes(env,10,opt_Policy)
        metrics[epsilon_val]['reward'] = total_reward
        metrics[epsilon_val]['iteration'] = converge_iteration
        metrics[epsilon_val]['time'] = end - start
        metrics[epsilon_val]['avg_reward'] = avg_reward
        metrics[epsilon_val]['num_wins'] = win
    with open(f'markov_decision_process/charts/frozen_lake/{dir}/epsilon_metrics.json','w') as f:
        json.dump(metrics,f,indent=4)


def get_rewards(value_funcs,env):
    wins = []
    rewards = []
    avg_rewards = []
    for value_function in value_funcs:
        # intialize optimal policy
        optimal_policy = np.zeros(env.nS, dtype = 'int8')
        policy = update_policy(env,optimal_policy,value_function,0.9)
        win, total_reward, avg_reward = play_episodes(env,10,policy)
        wins.append(win)
        rewards.append(total_reward)
        avg_rewards.append(avg_reward)
    return wins,rewards,avg_rewards

def generate_plots(value_funcs,env,figname,dir):
    wins, rewards, avg_rewards = get_rewards(value_funcs,env)
    _,ax = plt.subplots(1)
    ax.plot(avg_rewards)
    ax.set_title('Average Reward')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Avg Reward')
    plt.savefig(f'markov_decision_process/charts/frozen_lake/{dir}/{figname}_avg_reward.png')
    plt.clf()

    _,ax = plt.subplots(1)
    ax.plot(rewards)
    ax.set_title('Reward')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Reward')
    plt.savefig(f'markov_decision_process/charts/frozen_lake/{dir}/{figname}_reward.png')
    plt.clf()

    _,ax = plt.subplots(1)
    ax.plot(wins)
    ax.set_title('Wins')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('wins')
    plt.savefig(f'markov_decision_process/charts/frozen_lake/{dir}/{figname}_wins.png')
    plt.clf()




if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    env.seed(50)
    # start = time.time()
    # opt_V, opt_Policy, converge_iteration, value_funcs = value_iteration(env, max_iteration = 1000)
    # end = time.time()
    # elapsed_time = (end - start) * 1000
    # print (f"Time to converge: {elapsed_time: 0.3} ms")
    # print(f'coverged at iteration: {converge_iteration}')
    # print('Optimal Value function: ')
    # print(opt_V.reshape((4, 4)))
    # print('Final Policy: ')
    # print(opt_Policy)
    discount_experiment(env,'vi/gamma')
    epsilon_experiment(env,'vi/epsilon')