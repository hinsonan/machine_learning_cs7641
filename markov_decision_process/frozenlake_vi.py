import numpy as np
import gym
import gym.spaces as spaces
import time
from vi_pi_helpers import value_iteration, update_policy, play_episodes
import matplotlib.pyplot as plt

def discount_experiment():
    gamma_vals = [0.2,0.3,0.5,0.8,0.9]

def get_rewards(value_funcs,env):
    wins = []
    rewards = []
    avg_rewards = []
    for value_function in value_funcs:
        # intialize optimal policy
        optimal_policy = np.zeros(env.nS, dtype = 'int8')
        policy = update_policy(env,optimal_policy,value_function,0.9)
        win, total_reward, avg_reward = play_episodes(env,100,policy)
        wins.append(win)
        rewards.append(total_reward)
        avg_rewards.append(avg_rewards)
    return wins,rewards,avg_rewards

def generate_plots(value_funcs,env):
    wins, rewards, avg_rewards = get_rewards(value_funcs,env)
    _,ax = plt.subplots(1)
    ax.plot(rewards)
    ax.set_title('Rewards vs Iterations')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Reward')
    plt.show()


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
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
    opt_V, opt_Policy, converge_iteration, value_funcs = value_iteration(env, max_iteration = 1000)
    generate_plots(value_funcs,env)