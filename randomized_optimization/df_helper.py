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

def correct_GA_plots(curve_path, mutation_rate,problem_name):
    curve = pd.read_csv(curve_path)
    curve = curve.loc[(curve['Mutation Rate'] == mutation_rate)]

    curve.reset_index(inplace=True)

    ax = curve.Fitness.plot(title='Best Fitness GA')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig(f"randomized_optimization/corrected_plots/GA_{problem_name}")
    plt.clf()

def correct_RHC_plots(curve_path, restart, problem_name):
    curve = pd.read_csv(curve_path)
    curve = curve.loc[(curve['Restarts'] == restart)]

    curve.reset_index(inplace=True)

    ax = curve.Fitness.plot(title='Best Fitness RHC')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig(f"randomized_optimization/corrected_plots/RHC_{problem_name}")
    plt.clf()

def correct_MIMIC_plots(curve_path, keep, problem_name):
    curve = pd.read_csv(curve_path)
    curve = curve.loc[(curve['Keep Percent'] == keep)]

    curve.reset_index(inplace=True)

    ax = curve.Fitness.plot(title='Best Fitness MIMIC')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig(f"randomized_optimization/corrected_plots/MIMIC_{problem_name}")
    plt.clf()

correct_GA_plots('randomized_optimization/knapsack_csv/GA_best_curve.csv',0.5,'knapsack')
correct_GA_plots('randomized_optimization/n_queens_csv/GA_best_curve.csv',0.3,'queens')
correct_GA_plots('randomized_optimization/flipflop_csv/GA_best_curve.csv',0.6,'flipflop')

correct_RHC_plots('randomized_optimization/knapsack_csv/RHC_best_curve.csv',50,'knapsack')
correct_RHC_plots('randomized_optimization/n_queens_csv/RHC_best_curve.csv',100,'queens')
correct_RHC_plots('randomized_optimization/flipflop_csv/RHC_best_curve.csv',10,'flipflop')

correct_MIMIC_plots('randomized_optimization/knapsack_csv/MM_best_curve.csv',0.25,'knapsack')
correct_MIMIC_plots('randomized_optimization/n_queens_csv/MM_best_curve.csv',0.2,'queens')
correct_MIMIC_plots('randomized_optimization/flipflop_csv/MM_best_curve.csv',0.2,'flipflop')

