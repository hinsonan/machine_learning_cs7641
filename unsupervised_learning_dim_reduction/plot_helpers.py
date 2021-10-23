import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D

def plot_elbow(distances: dict, fig_name, dir='original_data'):
    _, axes = plt.subplots(1)
    axes.plot(list(distances.keys()), list(distances.values()))
    axes.set_title('KMeans Elbow Method')
    axes.set_xlabel('Number of Clusters')
    axes.set_ylabel('Sum of Squared Distances')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()

def plot_silhouette(silhoutte_scores: dict, fig_name:str, plot_title:str,dir='original_data'):
    _, axes = plt.subplots(1)
    axes.plot(list(silhoutte_scores.keys()), list(silhoutte_scores.values()))
    axes.set_title(plot_title)
    axes.set_xlabel('Number of Clusters')
    axes.set_ylabel('Silhoutte Score')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()

def plot_homo_and_complete(homo_scores:dict,completeness_scores:dict,fig_name:str,plot_title:str,dir='original_data'):
    _, axes = plt.subplots(1)
    axes.plot(list(homo_scores.keys()), list(homo_scores.values()))
    axes.plot(list(completeness_scores.keys()), list(completeness_scores.values()))
    axes.set_title(plot_title)
    axes.set_xlabel('Number of Clusters')
    axes.set_ylabel('Score')
    axes.legend(['homogeneity','completeness'])
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()

def pair_wise_plot(dataframe,label_name,fig_name,dir='original_data'):
    sb.set_style("ticks")
    sb.pairplot(dataframe,hue = label_name,diag_kind = "kde",kind = "scatter",palette = "husl")
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()

def plot_2d(df,fig_name,target_name,dir='pca_data'):
    sb.scatterplot(
    x=0, y=1,
    hue=target_name,
    palette=sb.color_palette("hls", 2),
    data=df,
    legend="full",
    )
    plt.title("PCA Reduction")
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()

def plot_3d(df,fig_name,target_name,dir='pca_data'):
    fig = plt.figure()
    ax = Axes3D(fig)

    sequence_containing_x_vals = df[0]
    sequence_containing_y_vals = df[1]
    sequence_containing_z_vals = df[2]

    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals,c=df[target_name])
    ax.set_title("PCA Reduction")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()

def plot_kurtosis(kurtosis_values:dict,fig_name,dir='ica_data'):
    _, axes = plt.subplots(1)
    axes.plot(list(kurtosis_values.keys()),list(kurtosis_values.values()))
    axes.set_title('Kurtosis')
    axes.set_xlabel('Number of Components')
    axes.set_ylabel('Kurtosis Value')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()

def plot_reconstruction_error(error_values:dict,fig_name,dir):
    _, axes = plt.subplots(1)
    axes.plot(list(error_values.keys()),list(error_values.values()))
    axes.set_title('Reconstruction Error')
    axes.set_xlabel('Number of Components')
    axes.set_ylabel('Error Value')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()

def plot_explained_variance(explained_variances:dict,fig_name,dir='pca_data'):
    _, axes = plt.subplots(1)
    axes.plot(list(explained_variances.keys()),list(explained_variances.values()))
    axes.set_title('Explained Variance Ratio')
    axes.set_xlabel('Number of Components')
    axes.set_ylabel('Value')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()
