import gym
import numpy as np

DISCOUNT = 1.0
STEP_REWARD = 0.0
LOSE_REWARD = 0.0
WIN_REWARD = 1.0

def avg_reward(env, s, a):
    avg_reward = 0
    for prob, next_s, reward, done in env.P[s][a]:
        if not done:
            avg_reward += prob * STEP_REWARD
        elif reward == 0.0:
#             avg_reward += prob * (-5)
            avg_reward += prob * LOSE_REWARD
        else:
#             avg_reward += prob * 10
            avg_reward += prob * WIN_REWARD
    return avg_reward
            
def random_policy(env):
    return np.random.randint(0, 4, size=env.nS)

def one_step_lookahead(env, s, value_function):
    action_values = np.zeros(env.nA)
    for a in range(env.nA):
        value = avg_reward(env, s, a)
        for p, next_s, _, _ in env.P[s][a]:
            value += DISCOUNT * p * value_function[next_s]
        action_values[a] = value
    return action_values
    
def evaluate_policy(env, policy, max_backups=1000, tol=1e-6):
    old_value = np.zeros(env.nS)
    for i in range(max_backups):
        new_value = np.zeros(env.nS)
        for s in range(env.nS):
            action_values = one_step_lookahead(env, s, old_value)
            new_value[s] = action_values[policy[s]]
        if np.max(np.abs(new_value-old_value)) < tol:
            break
        old_value = new_value
    return new_value

def greedy_policy(env, value_function):
    policy = np.zeros(env.nS, dtype=np.int32)
    for s in range(env.nS):
        action_values = one_step_lookahead(env, s, value_function)
        policy[s] = np.argmax(action_values)
    return policy

def policy_iteration(env, max_steps=100):
    old_policy = random_policy(env)
    for i in range(max_steps):
        value_function = evaluate_policy(env, old_policy)
        new_policy = greedy_policy(env, value_function)
        
        if np.array_equal(new_policy, old_policy):
            break
        old_policy = new_policy
    return old_policy, value_function

def main():
    env = gym.make('FrozenLake-v1')
    opt_policy, opt_value_func = policy_iteration(env)
    av_reward = []
    for i_episode in range(100):
        observation = env.reset()
        for t in range(10000):
    #         env.render()
            action = opt_policy[observation]
            observation, reward, done, info = env.step(action)
            if done:
                if reward == 0.0:
                    print("LOSE")
                else:
                    print("WIN")
                print("Episode finished after {} timesteps".format(t+1))
                break
        av_reward.append(reward)
        if i_episode % 1000 == 0:
            print('Current avg_reward: %f' % np.mean(av_reward))
    print(np.mean(av_reward))


if __name__ == "__main__":
    main()