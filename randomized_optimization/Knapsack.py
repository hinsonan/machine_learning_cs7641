import mlrose_hiive
import matplotlib.pyplot as plt
from df_helper import compute_avgs, compute_best_values
import numpy as np

def simulated_annealing_runner(problem):
    sa = mlrose_hiive.SARunner(problem, experiment_name="SA_knapsack", iteration_list=[1000],
                                temperature_list=[1, 10, 50, 100, 300, 500, 1000, 1500, 2000],
                                decay_list=[mlrose_hiive.ExpDecay,
                                        mlrose_hiive.GeomDecay],
                            seed=44, max_attempts=100)

    sa_run_stats, sa_run_curves = sa.run()

    print('*************SA BEGIN*************')

    final_iter = sa_run_stats[sa_run_stats.Iteration != 0].reset_index()
    print('***********BEST FITNESS**********')
    compute_best_values(final_iter)

    print('**********TIMES*************')
    compute_avgs(final_iter, 'schedule_init_temp',[1, 10, 50, 100, 300, 500, 1000, 1500, 2000],'Time')
    print(f'Time Avg: {final_iter.loc[:, "Time"].mean()}')

    print('*************Function Evals*************')
    compute_avgs(final_iter, 'schedule_init_temp',[1, 10, 50, 100, 300, 500, 1000, 1500, 2000],'FEvals')
    print(f'FEval Avg: {final_iter.loc[:, "FEvals"].mean()}')

    best_index_in_curve = sa_run_curves.Fitness.idxmax()
    best_decay = sa_run_curves.iloc[best_index_in_curve].Temperature
    best_curve = sa_run_curves.loc[sa_run_curves.Temperature == best_decay, :]
    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(colormap='jet', marker='.', markersize=2, 
                                figsize=(12,8), grid=1,
                                title='Best Simulated Annealing')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/SA_knapsack")

def hill_climbing_runner(problem):
    rhc = mlrose_hiive.RHCRunner(problem, experiment_name="RHC_knapsack", 
                                        iteration_list=[1000],
                                        seed=44, max_attempts=100, 
                                        restart_list=[100])
    rhc_run_stats, rhc_run_curves = rhc.run()

    print('*************RHC BEGIN*************')

    final_iter = rhc_run_stats[rhc_run_stats.Iteration != 0].reset_index()
    print('***********BEST FITNESS**********')
    compute_best_values(final_iter)

    print('**********TIMES*************')
    compute_avgs(final_iter, 'Restarts', [100], 'Time')

    print('*************Function Evals*************')
    compute_avgs(final_iter, 'Restarts',[100],'FEvals')

    best_index_in_curve = rhc_run_curves.Fitness.idxmax()
    best_decay = rhc_run_curves.iloc[best_index_in_curve].current_restart
    best_curve = rhc_run_curves.loc[rhc_run_curves.current_restart == best_decay, :]
    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(colormap='jet', marker='.', markersize=2, 
                                figsize=(12,8), grid=1,
                                title='Best RHC')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/RHC_knapsack")

def genetic_algorithms(problem):
    ga = mlrose_hiive.GARunner(problem=problem,
                            experiment_name="GA_Exp",
                            seed=44,
                            iteration_list=[1000],
                            max_attempts=100,
                            population_sizes=[50, 200, 500],
                            mutation_rates=[0.1, 0.25, 0.5])
    ga_run_stats, ga_run_curves = ga.run()

    print('*************GA BEGIN*************')

    final_iter = ga_run_stats[ga_run_stats.Iteration != 0].reset_index()
    print('***********BEST FITNESS**********')
    compute_best_values(final_iter)

    print('**********TIMES*************')
    compute_avgs(final_iter, 'Population Size', [50,200,500], 'Time')
    print(f'Time Avg: {final_iter.loc[:, "Time"].mean()}')

    print('*************Function Evals*************')
    compute_avgs(final_iter, 'Population Size', [50,200,500], 'FEvals')
    print(f'FEval Avg: {final_iter.loc[:, "FEvals"].mean()}')

    best_index_in_curve = ga_run_curves.Fitness.idxmax()
    best_decay = ga_run_curves.iloc[best_index_in_curve]["Population Size"]
    best_curve = ga_run_curves.loc[ga_run_curves["Population Size"] == best_decay, :]
    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(colormap='jet', marker='.', markersize=2, 
                                figsize=(12,8), grid=1,
                                title='Best GA')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/GA_knapsack")

def mimic_runner(problem):
    mmc = mlrose_hiive.MIMICRunner(problem=problem,
                    experiment_name="MMC_Exp",
                    seed=44,
                    iteration_list=[1000],
                    max_attempts=100,
                    population_sizes=[50, 200, 500],
                    keep_percent_list=[0.25, 0.5, 0.75],
                    use_fast_mimic=True)

    # the two data frames will contain the results
    mmc_run_stats, mmc_run_curves = mmc.run()

    print('*************MIMIC BEGIN*************')

    final_iter = mmc_run_stats[mmc_run_stats.Iteration != 0].reset_index()
    print('***********BEST FITNESS**********')
    compute_best_values(final_iter)

    print('**********TIMES*************')
    compute_avgs(final_iter, 'Population Size', [50,200,500], 'Time')
    print(f'Time Avg: {final_iter.loc[:, "Time"].mean()}')

    print('*************Function Evals*************')
    compute_avgs(final_iter, 'Population Size', [50,200,500], 'FEvals')
    print(f'FEval Avg: {final_iter.loc[:, "FEvals"].mean()}')

    best_index_in_curve = mmc_run_curves.Fitness.idxmax()
    best_decay = mmc_run_curves.iloc[best_index_in_curve]["Population Size"]
    best_curve = mmc_run_curves.loc[mmc_run_curves["Population Size"] == best_decay, :]
    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(colormap='jet', marker='.', markersize=2, 
                                figsize=(12,8), grid=1,
                                title='Best MM')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/MM_knapsack")

knapsack_len = 200

knapsack_weights=np.random.uniform(10,60,knapsack_len)

knapsack_values=np.random.uniform(10,40,knapsack_len)

problem = mlrose_hiive.KnapsackOpt(weights=knapsack_weights,values=knapsack_values,max_weight_pct=.5)

simulated_annealing_runner(problem)

hill_climbing_runner(problem)

genetic_algorithms(problem)

mimic_runner(problem)
