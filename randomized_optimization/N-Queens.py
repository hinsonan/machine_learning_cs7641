import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
def simulated_annealing_runner(problem):
    sa = mlrose_hiive.SARunner(problem, experiment_name="SA_QUEENS", iteration_list=[1000],
                                temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000],
                                decay_list=[mlrose_hiive.ExpDecay,
                                        mlrose_hiive.GeomDecay],
                            seed=44, max_attempts=100)

    sa_run_stats, sa_run_curves = sa.run()

    best_index_in_curve = sa_run_curves.Fitness.idxmax()
    best_decay = sa_run_curves.iloc[best_index_in_curve].Temperature
    best_curve = sa_run_curves.loc[sa_run_curves.Temperature == best_decay, :]
    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(lw=2, colormap='jet', marker='.', markersize=2, 
                                figsize=(12,8), grid=1,
                                title='Best Simulated Annealing')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/SA_queens")

def hill_climbing_runner(problem):
    rhc = mlrose_hiive.RHCRunner(problem, experiment_name="RHC_QUEENS", 
                                        iteration_list=[10000],
                                        seed=44, max_attempts=100, 
                                        restart_list=[100])
    rhc_run_stats, rhc_run_curves = rhc.run()

    best_index_in_curve = rhc_run_curves.Fitness.idxmax()
    best_decay = rhc_run_curves.iloc[best_index_in_curve].current_restart
    best_curve = rhc_run_curves.loc[rhc_run_curves.current_restart == best_decay, :]
    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(lw=2, colormap='jet', marker='.', markersize=2, 
                                figsize=(12,8), grid=1,
                                title='Best RHC')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/RHC_queens")

def genetic_algorithms(problem):
    ga = mlrose_hiive.GARunner(problem=problem,
                            experiment_name="GA_Exp",
                            seed=44,
                            iteration_list=[100000],
                            max_attempts=100,
                            population_sizes=[50, 200, 500],
                            mutation_rates=[0.1, 0.25, 0.5])
    ga_run_stats, ga_run_curves = ga.run()

    best_index_in_curve = ga_run_curves.Fitness.idxmax()
    best_decay = ga_run_curves.iloc[best_index_in_curve]["Population Size"]
    best_curve = ga_run_curves.loc[ga_run_curves["Population Size"] == best_decay, :]
    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(lw=2, colormap='jet', marker='.', markersize=2, 
                                figsize=(12,8), grid=1,
                                title='Best GA')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/GA_queens")

def mimic_runner(problem):
    mmc = mlrose_hiive.MIMICRunner(problem=problem,
                    experiment_name="MMC_Exp",
                    seed=44,
                    iteration_list=[100000],
                    max_attempts=100,
                    population_sizes=[50, 200, 500],
                    keep_percent_list=[0.25, 0.5, 0.75],
                    use_fast_mimic=True)

    # the two data frames will contain the results
    mmc_run_stats, mmc_run_curves = mmc.run()

    best_index_in_curve = mmc_run_curves.Fitness.idxmax()
    best_decay = mmc_run_curves.iloc[best_index_in_curve]["Population Size"]
    best_curve = mmc_run_curves.loc[mmc_run_curves["Population Size"] == best_decay, :]
    best_curve.reset_index(inplace=True)

    ax = best_curve.Fitness.plot(lw=2, colormap='jet', marker='.', markersize=2, 
                                figsize=(12,8), grid=1,
                                title='Best MM')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    plt.savefig("randomized_optimization/MM_queens")

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):

   # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt

fitness = mlrose_hiive.CustomFitness(queens_max)

problem = mlrose_hiive.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)

schedule = mlrose_hiive.ExpDecay()

# Solve using simulated annealing - attempt 1         
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(problem, schedule = schedule, max_attempts = 10, 
                                                      max_iters = 1000, init_state = init_state,
                                                      random_state = 1)

print('The best state found is: ', best_state)

print('The fitness at the best state is: ', best_fitness)


simulated_annealing_runner(problem)

hill_climbing_runner(problem)

genetic_algorithms(problem)

mimic_runner(problem)
