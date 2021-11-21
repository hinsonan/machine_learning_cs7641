from hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
from numpy.random import choice
import matplotlib.pyplot as plt
import json

def plot_results(result_dic,experiment_name,dir='vi/gamma'):
    for key in result_dic.keys():
        for key2, vals in result_dic[key].items():
            # dont plot the iterations
            if key2 == 'iteration':
                continue
            _,ax = plt.subplots(1)
            ax.plot(vals)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(f'{key2}')
            ax.set_title(f'{key2} vs Iterations')
            fig_name = f'{experiment_name}_{key}_{key2}'
            plt.savefig(f"markov_decision_process/charts/{dir}/{fig_name}.png")
            plt.clf()

def write_results(result_dic,experiment_name,dir='vi/gamma'):
    output = {}
    for key in result_dic.keys():
        output[key] = {"max_reward":None, "max_avg_reward":None,"number_of_iterations":None,"time":None}
        for key2, vals in result_dic[key].items():
            if key2 == 'reward':
                output[key]['max_reward'] = max(vals)
            elif key2 == 'iteration':
                output[key]['number_of_iterations'] = max(vals)
            elif key2 == 'time':
                output[key]['time'] = max(vals)
            else:
                output[key]['max_avg_reward'] = max(vals)
    with open(f'markov_decision_process/charts/{dir}/{experiment_name}_metrics.json','w') as f:
        json.dump(output,f,indent=4)

def vi_gamma_experiment(P,R,prob_size_dir):
    gamma_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    result_dic = {}
    for gamma_val in gamma_list:
        result_dic[gamma_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[]}
        vi = ValueIteration(P,R,gamma=gamma_val,epsilon=0.01)
        vi.run()
        stats = vi.run_stats
        for dic in stats:
            result_dic[gamma_val]['reward'].append(dic['Reward'])
            result_dic[gamma_val]['iteration'].append(dic['Iteration'])
            result_dic[gamma_val]['time'].append(dic['Time'])
            result_dic[gamma_val]['avg_reward'].append(dic['Mean V'])
    plot_results(result_dic,'gamma',dir=f'{prob_size_dir}/vi/gamma')
    write_results(result_dic,'gamma',dir=f'{prob_size_dir}/vi/gamma')

def pi_gamma_experiment(P,R,prob_size_dir):
    gamma_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    result_dic = {}
    for gamma_val in gamma_list:
        result_dic[gamma_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[]}
        vi = PolicyIteration(P,R,gamma=gamma_val)
        vi.run()
        stats = vi.run_stats
        for dic in stats:
            result_dic[gamma_val]['reward'].append(dic['Reward'])
            result_dic[gamma_val]['iteration'].append(dic['Iteration'])
            result_dic[gamma_val]['time'].append(dic['Time'])
            result_dic[gamma_val]['avg_reward'].append(dic['Mean V'])
    plot_results(result_dic,'gamma',dir=f'{prob_size_dir}/pi/gamma')
    write_results(result_dic,'gamma',dir=f'{prob_size_dir}/pi/gamma')

def qlearner_gamma_experiment(P,R,prob_size_dir):
    gamma_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    result_dic = {}
    for gamma_val in gamma_list:
        result_dic[gamma_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[]}
        vi = QLearning(P,R,gamma_val)
        vi.run()
        stats = vi.run_stats
        for dic in stats:
            result_dic[gamma_val]['reward'].append(dic['Reward'])
            result_dic[gamma_val]['iteration'].append(dic['Iteration'])
            result_dic[gamma_val]['time'].append(dic['Time'])
            result_dic[gamma_val]['avg_reward'].append(dic['Mean V'])
    plot_results(result_dic,'gamma',dir=f'{prob_size_dir}/qlearner/gamma')
    write_results(result_dic,'gamma',dir=f'{prob_size_dir}/qlearner/gamma')

def vi_epsilon_experiment(P,R,prob_size_dir):
    epsilon_list = [1,0.5,0.1,0.01,0.001,0.0001]
    result_dic = {}
    for epsilon_val in epsilon_list:
        result_dic[epsilon_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[]}
        vi = ValueIteration(P,R,gamma=0.9,epsilon=epsilon_val)
        vi.run()
        stats = vi.run_stats
        for dic in stats:
            result_dic[epsilon_val]['reward'].append(dic['Reward'])
            result_dic[epsilon_val]['iteration'].append(dic['Iteration'])
            result_dic[epsilon_val]['time'].append(dic['Time'])
            result_dic[epsilon_val]['avg_reward'].append(dic['Mean V'])
    plot_results(result_dic,'epsilon',dir=f'{prob_size_dir}/vi/epsilon')
    write_results(result_dic,'epsilon',dir=f'{prob_size_dir}/vi/epsilon')

def qlearner_epsilon_experiment(P,R,prob_size_dir):
    epsilon_list = [1,10,50,100,150]
    result_dic = {}
    for epsilon_val in epsilon_list:
        result_dic[epsilon_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[]}
        vi = QLearning(P,R,0.9,epsilon=epsilon_val)
        vi.run()
        stats = vi.run_stats
        for dic in stats:
            result_dic[epsilon_val]['reward'].append(dic['Reward'])
            result_dic[epsilon_val]['iteration'].append(dic['Iteration'])
            result_dic[epsilon_val]['time'].append(dic['Time'])
            result_dic[epsilon_val]['avg_reward'].append(dic['Mean V'])
    plot_results(result_dic,'epsilon',dir=f'{prob_size_dir}/qlearner/epsilon')
    write_results(result_dic,'epsilon',dir=f'{prob_size_dir}/qlearner/epsilon')

def qlearner_alpha_experiment(P,R,prob_size_dir):
    alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    result_dic = {}
    for alpha_val in alpha_list:
        result_dic[alpha_val] = {'reward':[],'iteration':[],'time':[],'avg_reward':[]}
        vi = QLearning(P,R,0.9,alpha=alpha_val)
        vi.run()
        stats = vi.run_stats
        for dic in stats:
            result_dic[alpha_val]['reward'].append(dic['Reward'])
            result_dic[alpha_val]['iteration'].append(dic['Iteration'])
            result_dic[alpha_val]['time'].append(dic['Time'])
            result_dic[alpha_val]['avg_reward'].append(dic['Mean V'])
    plot_results(result_dic,'learning_rate',dir=f'{prob_size_dir}/qlearner/learning_rate')
    write_results(result_dic,'learning_rate',dir=f'{prob_size_dir}/qlearner/learning_rate')
    

if __name__ == '__main__':
    P, R = forest(S=150)
    # gamma experiments
    vi_gamma_experiment(P,R,'forest_small')
    pi_gamma_experiment(P,R, 'forest_small')
    qlearner_gamma_experiment(P,R,'forest_small')

    # epsilon experiments 
    # NOTE PI has no epsilon value
    vi_epsilon_experiment(P,R,'forest_small')
    qlearner_epsilon_experiment(P,R,'forest_small')
    qlearner_alpha_experiment(P,R,'forest_small')

    P, R = forest(S=2000)
    # gamma experiments
    vi_gamma_experiment(P,R,'forest_large')
    pi_gamma_experiment(P,R, 'forest_large')
    qlearner_gamma_experiment(P,R,'forest_large')

    # epsilon experiments 
    # NOTE PI has no epsilon value
    vi_epsilon_experiment(P,R,'forest_large')
    qlearner_epsilon_experiment(P,R,'forest_large')
    qlearner_alpha_experiment(P,R,'forest_large')

        