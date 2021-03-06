import mlrose_hiive
import matplotlib.pyplot as plt
from df_helper import compute_avgs, compute_best_values
import numpy as np
import time
import pandas as pd

def simulated_annealing_runner(problem):
    sa = mlrose_hiive.SARunner(problem, experiment_name="SA_flipflop", iteration_list=[1000],
                                temperature_list=[1,10,20,30,300],
                                decay_list=[mlrose_hiive.ExpDecay,mlrose_hiive.GeomDecay],
                            seed=64, max_attempts=100)
    start = time.time()
    sa_run_stats, sa_run_curves = sa.run()
    end = time.time()

    print('*************SA BEGIN*************')

    final_iter = sa_run_stats[sa_run_stats.Iteration != 0].reset_index()
    print('***********BEST FITNESS**********')
    compute_best_values(final_iter)

    print('*************Function Evals*************')
    compute_avgs(final_iter, 'schedule_init_temp',[1,10,20,30],'FEvals')
    print(f'FEval Avg: {final_iter.loc[:, "FEvals"].mean()}')

    best_idx = sa_run_curves.Fitness.idxmax()
    best_temp = sa_run_curves.iloc[best_idx].Temperature
    best_curve = sa_run_curves.loc[sa_run_curves.Temperature == best_temp, :]

    print('**********TIME FOR BEST CURVE*************')
    print(f'Temperature {best_temp} Time {best_curve.iloc[-1].Time}')

    print('**********WALL CLOCK TIME*************')
    print(f'Walk Clock Time: {end-start}')

    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(title='Best Fitness Simulated Annealing')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/SA_flipflop")
    plt.clf()

    final_iter.to_csv('randomized_optimization/flipflop_csv/SA_final_iter.csv',index=False, header=True)
    best_curve.to_csv('randomized_optimization/flipflop_csv/SA_best_curve.csv',index=False, header=True)

def hill_climbing_runner(problem):
    rhc = mlrose_hiive.RHCRunner(problem, experiment_name="RHC_flipflop", 
                                        iteration_list=[1000],
                                        seed=64, max_attempts=100, 
                                        restart_list=[10,30])
    start = time.time()
    rhc_run_stats, rhc_run_curves = rhc.run()
    end = time.time()

    print('*************RHC BEGIN*************')

    final_iter = rhc_run_stats[rhc_run_stats.Iteration != 0].reset_index()
    print('***********BEST FITNESS**********')
    compute_best_values(final_iter)

    print('*************Function Evals*************')
    compute_avgs(final_iter, 'Restarts',[50,100],'FEvals')

    best_idx = rhc_run_curves.Fitness.idxmax()
    best_restart = rhc_run_curves.iloc[best_idx].current_restart
    best_curve = rhc_run_curves.loc[rhc_run_curves.current_restart == best_restart, :]

    print('**********TIME FOR BEST CURVE*************')
    print(f'Restart {pd.unique(best_curve.Restarts)} Time {best_curve.iloc[-1].Time}')

    print('**********WALL CLOCK TIME*************')
    print(f'Walk Clock Time: {end-start}')

    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(title='Best Fitness Random Hill Climbing')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/RHC_flipflop")
    plt.clf()

    final_iter.to_csv('randomized_optimization/flipflop_csv/RHC_final_iter.csv',index=False, header=True)
    best_curve.to_csv('randomized_optimization/flipflop_csv/RHC_best_curve.csv',index=False, header=True)

def genetic_algorithms(problem):
    ga = mlrose_hiive.GARunner(problem=problem,
                            experiment_name="GA_Exp",
                            seed=64,
                            iteration_list=[1000],
                            max_attempts=100,
                            population_sizes=[50, 100, 200, 500],
                            mutation_rates=[0.1, 0.3, 0.6])
    start = time.time()
    ga_run_stats, ga_run_curves = ga.run()
    end = time.time()

    print('*************GA BEGIN*************')

    final_iter = ga_run_stats[ga_run_stats.Iteration != 0].reset_index()
    print('***********BEST FITNESS**********')
    compute_best_values(final_iter)

    print('*************Function Evals*************')
    compute_avgs(final_iter, 'Population Size', [50,100,200,500], 'FEvals')
    print(f'FEval Avg: {final_iter.loc[:, "FEvals"].mean()}')

    best_idx = ga_run_curves.Fitness.idxmax()
    best_population = ga_run_curves.iloc[best_idx]["Population Size"]
    best_curve = ga_run_curves.loc[ga_run_curves["Population Size"] == best_population, :]

    print('**********TIME FOR BEST CURVE*************')
    print(f'Population {best_population} Time {best_curve.iloc[-1].Time}')

    print('**********WALL CLOCK TIME*************')
    print(f'Walk Clock Time: {end-start}')

    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(title='Best Fitness Genetic Algorithm')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/GA_flipflop")
    plt.clf()

    final_iter.to_csv('randomized_optimization/flipflop_csv/GA_final_iter.csv',index=False, header=True)
    best_curve.to_csv('randomized_optimization/flipflop_csv/GA_best_curve.csv',index=False, header=True)

def mimic_runner(problem):
    mmc = mlrose_hiive.MIMICRunner(problem=problem,
                    experiment_name="MMC_Exp",
                    seed=64,
                    iteration_list=[1000],
                    max_attempts=100,
                    population_sizes=[50, 100, 200, 500],
                    keep_percent_list=[0.2, 0.5, 0.8],
                    use_fast_mimic=True)

    start = time.time()
    mmc_run_stats, mmc_run_curves = mmc.run()
    end = time.time()

    print('*************MIMIC BEGIN*************')

    final_iter = mmc_run_stats[mmc_run_stats.Iteration != 0].reset_index()
    print('***********BEST FITNESS**********')
    compute_best_values(final_iter)

    print('*************Function Evals*************')
    compute_avgs(final_iter, 'Population Size', [50,100,200,500], 'FEvals')
    print(f'FEval Avg: {final_iter.loc[:, "FEvals"].mean()}')

    best_idx = mmc_run_curves.Fitness.idxmax()
    best_population = mmc_run_curves.iloc[best_idx]["Population Size"]
    best_curve = mmc_run_curves.loc[mmc_run_curves["Population Size"] == best_population, :]

    print('**********TIME FOR BEST CURVE*************')
    print(f'Population {best_population} Time {best_curve.iloc[-1].Time}')

    print('**********WALL CLOCK TIME*************')
    print(f'Walk Clock Time: {end-start}')

    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(title='Best Fitness Mimic')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/MM_flipflop")
    plt.clf()

    final_iter.to_csv('randomized_optimization/flipflop_csv/MM_final_iter.csv',index=False, header=True)
    best_curve.to_csv('randomized_optimization/flipflop_csv/MM_best_curve.csv',index=False, header=True)

problem = mlrose_hiive.FlipFlopOpt(length=300)

# simulated_annealing_runner(problem)

hill_climbing_runner(problem)

# genetic_algorithms(problem)

# mimic_runner(problem)
