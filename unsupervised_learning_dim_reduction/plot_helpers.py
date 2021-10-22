import matplotlib.pyplot as plt
import seaborn as sb

def plot_elbow(distances: dict, fig_name):
    _, axes = plt.subplots(1)
    axes.plot(list(distances.keys()), list(distances.values()))
    axes.set_title('KMeans Elbow Method')
    axes.set_xlabel('Number of Clusters')
    axes.set_ylabel('Sum of Squared Distances')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{fig_name}')

def plot_silhouette(silhoutte_scores: dict, fig_name:str, plot_title:str):
    _, axes = plt.subplots(1)
    axes.plot(list(silhoutte_scores.keys()), list(silhoutte_scores.values()))
    axes.set_title(plot_title)
    axes.set_xlabel('Number of Clusters')
    axes.set_ylabel('Silhoutte Score')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{fig_name}')

def plot_homo_and_complete(homo_scores:dict,completeness_scores:dict,fig_name:str,plot_title:str):
    _, axes = plt.subplots(1)
    axes.plot(list(homo_scores.keys()), list(homo_scores.values()))
    axes.plot(list(completeness_scores.keys()), list(completeness_scores.values()))
    axes.set_title(plot_title)
    axes.set_xlabel('Number of Clusters')
    axes.set_ylabel('Score')
    axes.legend(['homogeneity','completeness'])
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{fig_name}')

def pair_wise_plot(dataframe,label_name,fig_name):
    sb.set_style("ticks")
    sb.pairplot(dataframe,hue = label_name,diag_kind = "kde",kind = "scatter",palette = "husl")
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{fig_name}')
    plt.clf()