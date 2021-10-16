import matplotlib.pyplot as plt
import pandas as pd

def compute_avgs(dataframe, column, params, avg_column):
    dic = {}
    for param in params:
        avg = dataframe.loc[dataframe[column] == param, avg_column].mean()
        dic[param] = avg
    for key, item in dic.items():
        print(f'{key}: {item}')

def compute_best_values(dataframe):
    print(f'Fitness Avg: {dataframe.Fitness.mean()}')
    print(f'Fitness Max: {dataframe.Fitness.max()}')

def plot_evaluations(csv_paths,problem):
    for path in csv_paths:
        dataframe = pd.read_csv(path)
        path = path.split('/')[2]
        if path[0] == 'R':
            filename = f'{path[:3]}_{problem}_Evals'
        else:
            filename = f'{path[:2]}_{problem}_Evals'
        ax = dataframe.FEvals.plot(title='Number of Function Evaluations')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Value")
        plt.savefig(f"randomized_optimization/{filename}")
        plt.clf()
