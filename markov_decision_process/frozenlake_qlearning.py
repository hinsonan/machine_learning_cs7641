import numpy as np
import gym
import random
import pandas as pd
import matplotlib.pyplot as plt
import json

class QLearner():

    random.seed(2)

    def __init__(self,env_actions,env_states, number_iterations=10000) -> None:
        self.action_size = env_actions
        self.state_size = env_states

        self.qtable = np.zeros((self.state_size, self.action_size))

        self.total_episodes = number_iterations        # Total episodes
        self.learning_rate = 0.8           # Learning rate
        self.max_steps = 99                # Max steps per episode
        self.gamma = 0.95                  # Discounting rate

        # Exploration parameters
        self.epsilon = 1.0                 # Exploration rate
        self.max_epsilon = 1.0             # Exploration probability at start
        self.min_epsilon = 0.01            # Minimum exploration probability 
        self.decay_rate = 0.005             # Exponential decay rate for exploration prob

        # List of rewards
        self.rewards = []

        # plotting variables
        self.df = pd.DataFrame(columns=["Iteration", "Reward"])

    def plot_df(self,title,value):
        self.df.plot(x='Iteration',y="Reward")
        plt.savefig(f'markov_decision_process/charts/{title}_{value}_qlearner')

    def train(self,env):

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

        
        self.plot_df('iterations_vs_reward',self.total_episodes)

    def evaluate_policy(self, env, fig_name, value):
        wins = 0
        losses = 0
        steps_per_game = []

        for episode in range(101):
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
        
        dic = {'avg_steps_per_game': sum(steps_per_game)/len(steps_per_game),'wins':wins,'losses':losses}
        with open(f'markov_decision_process/charts/{fig_name}_{value}_eval','w') as f:
            json.dump(dic,f,indent=4)

def iteration_experiment(env, iteration_list=[500,1500,5000,10000,12000]):
    for iterations in iteration_list:
        qlearner  = QLearner(env.action_space.n,env.observation_space.n,iterations)
        qlearner.train(env)
        # EVALUATE QTABLE
        qlearner.evaluate_policy(env,"iterations_vs_reward",qlearner.total_episodes)
        env.close()
        

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    env.seed(50)

    iteration_experiment(env)