import matplotlib.pyplot as plt

def plot_elbow(distances: dict, fig_name):
    _, axes = plt.subplots(1)
    axes.plot(list(distances.keys()), list(distances.values()))
    axes.set_title('KMeans Elbow Method')
    axes.set_xlabel('Number of Clusters')
    axes.set_ylabel('Sum of Squared Distances')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{fig_name}')

def plot_silhouette(silhoutte_scores: dict, fig_name):
    _, axes = plt.subplots(1)
    axes.plot(list(silhoutte_scores.keys()), list(silhoutte_scores.values()))
    axes.set_title('KMeans Silhouette Score')
    axes.set_xlabel('Number of Clusters')
    axes.set_ylabel('Silhoutte Score')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{fig_name}')