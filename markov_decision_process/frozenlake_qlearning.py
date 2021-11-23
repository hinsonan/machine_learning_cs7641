import numpy as np
import gym
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv

class QLearner():

    random.seed(2)

    def __init__(self,env_actions,env_states, number_iterations=10000,lr=0.8,epsilon=1.0) -> None:
        self.action_size = env_actions
        self.state_size = env_states

        self.qtable = np.zeros((self.state_size, self.action_size))

        self.total_episodes = number_iterations        # Total episodes
        self.learning_rate = lr           # Learning rate
        self.max_steps = 99                # Max steps per episode
        self.gamma = 0.95                  # Discounting rate

        # Exploration parameters
        self.epsilon = epsilon                 # Exploration rate
        self.max_epsilon = epsilon             # Exploration probability at start
        self.min_epsilon = 0.01            # Minimum exploration probability 
        self.decay_rate = 0.005             # Exponential decay rate for exploration prob

        # List of rewards
        self.rewards = []

        # plotting variables
        self.df = pd.DataFrame(columns=["Iteration", "Reward"])

    def plot_df(self,title,value,experiment_name,size_dir='frozen_lake_small'):
        self.df.plot(x='Iteration',y="Reward")
        plt.savefig(f'markov_decision_process/charts/{size_dir}/qlearner/{title}_{value}_{experiment_name}.png')

    def train(self,env,experiment_name,value,size_dir='frozen_lake_small'):

        # 2 For life or until learning is stopped
        for episode in range(self.total_episodes):
            # Reset the environment
            state = env.reset()
            step = 0
            done = False
            total_rewards = 0
            print(f'On Episode {episode}')
            for step in range(self.max_steps):
                # 3. Choose an action a in the current world state (s)
                ## First we randomize a number
                exp_exp_tradeoff = random.uniform(0, 1)
                
                ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
                if exp_exp_tradeoff > self.epsilon:
                    action = np.argmax(self.qtable[state,:])

                # Else doing a random choice --> exploration
                else:
                    action = env.action_space.sample()

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = env.step(action)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                # qtable[new_state,:] : all the actions we can take from new state
                self.qtable[state, action] = self.qtable[state, action] + self.learning_rate * (reward + self.gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])
                
                total_rewards += reward
                
                # Our new state is state
                state = new_state
                
                # If done (if we're dead) : finish episode
                if done == True: 
                    break
                
            # Reduce epsilon (because we need less and less exploration)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*episode) 
            self.rewards.append(total_rewards)
            # add the reward and iteration to the plotting df
            if self.df['Reward'].shape[0] == 0:
                self.df = self.df.append({'Iteration':episode,'Reward':total_rewards}, ignore_index=True)
            else:
                self.df = self.df.append({'Iteration':episode,'Reward':self.df['Reward'].iloc[-1]+total_rewards}, ignore_index=True)

        
        self.plot_df('iterations_vs_reward',value,experiment_name,size_dir)

    def evaluate_policy(self, env):
        wins = 0
        losses = 0
        steps_per_game = []

        for episode in range(10):
            state = env.reset()
            step = 0
            done = False
            print("****************************************************")
            print("EPISODE ", episode)

            for step in range(self.max_steps):
                
                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(self.qtable[state,:])
                
                new_state, reward, done, info = env.step(action)
                
                if done:
                    # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                    env.render()
                    
                    # We print the number of step it took.
                    print("Number of steps", step)

                    # gather metrics for this episode
                    if reward == 1:
                        wins += 1
                    else:
                        losses +=1
                    steps_per_game.append(step)
                    break
                state = new_state
        
        dic = {'avg_steps_per_game': sum(steps_per_game)/len(steps_per_game),'wins':wins,'losses':losses, 'reward':self.df['Reward'].values[-1]}
        return dic

def iteration_experiment(env, iteration_list=[500,1500,5000,10000,12000],size_dir='frozen_lake_small'):
    dic = {}
    for iterations in iteration_list:
        qlearner  = QLearner(env.action_space.n,env.observation_space.n,number_iterations=iterations)
        qlearner.train(env,'iterations',qlearner.total_episodes,size_dir=size_dir)
        # EVALUATE QTABLE
        result_dic = qlearner.evaluate_policy(env)
        dic[iterations] = result_dic
        env.close()
    with open(f'markov_decision_process/charts/{size_dir}/qlearner/iterations_eval.json','w') as f:
        json.dump(dic,f,indent=4)

def learning_rate_experiment(env, learning_rates=[0.001,0.01,0.1,0.2,0.9],iterations=10000,size_dir='frozen_lake_small'):
    dic = {}
    for lr in learning_rates:
        qlearner  = QLearner(env.action_space.n,env.observation_space.n,number_iterations=iterations,lr=lr)
        qlearner.train(env, 'learning_rate', qlearner.learning_rate,size_dir=size_dir)
        # EVALUATE QTABLE
        result_dic = qlearner.evaluate_policy(env)
        dic[lr] = result_dic
        env.close()
    with open(f'markov_decision_process/charts/{size_dir}/qlearner/learning_rate_eval.json','w') as f:
        json.dump(dic,f,indent=4)

def exploration_experiment(env, epsilons=[1.0, 10.0, 20.0, 30.0],iterations=10000,size_dir='frozen_lake_small'):
    dic = {}
    for epsilon in epsilons:
        qlearner  = QLearner(env.action_space.n,env.observation_space.n,number_iterations=iterations,epsilon=epsilon)
        qlearner.train(env, 'Epsilon', qlearner.epsilon,size_dir=size_dir)
        # EVALUATE QTABLE
        result_dic = qlearner.evaluate_policy(env)
        dic[epsilon] = result_dic
        env.close()
    with open(f'markov_decision_process/charts/{size_dir}/qlearner/exploration_eval.json','w') as f:
        json.dump(dic,f,indent=4)
        

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    env.seed(50)

    # iteration_experiment(env)
    learning_rate_experiment(env)
    # exploration_experiment(env)

    np.random.seed(10)
    env = FrozenLakeEnv(generate_random_map(20))

    # iteration_experiment(env,iteration_list=[100000],size_dir='frozen_lake_large')
    # learning_rate_experiment(env,learning_rates=[0.1,0.5,0.9],iterations=30000,size_dir='frozen_lake_large')
    # exploration_experiment(env,iterations=30000,size_dir='frozen_lake_large')