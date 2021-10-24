from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from data_helpers import get_cs_go_data, get_loan_defualt
from plot_helpers import plot_2d, plot_3d, plot_reconstruction_error,plot_explained_variance
from clustering import kmeans_experiment,gmm_experiment,evaluate_gmm,evaluate_kmeans
import numpy as np
import pandas as pd

def pca_reconstruction_metrics(data,range,dataset):
    explained_variances = {}
    errors = {}
    for i in range:
        pca = PCA(n_components=i)
        reduced_data = pca.fit_transform(data)

        # reconstruct the data and get the error
        reconstructed_data = np.dot(reduced_data,pca.components_)
        explained_variance = pca.explained_variance_ratio_
        error = mean_squared_error(data,reconstructed_data)
        explained_variances[i] = sum(explained_variance)
        errors[i] = error
        print(f'Range: {i} Explained Variance Ratio: {sum(explained_variance)}, Reconstruction Error: {error}')
    plot_reconstruction_error(errors,f'{dataset}_reconstruction','pca_data')
    plot_explained_variance(explained_variances,f'{dataset}_explained_varaince')

def pca_experiment(data,labels,num_dim,dataset_name):
    pca = PCA(n_components=num_dim)
    reduced_data = pca.fit_transform(data)

    # reconstruct the data and get the error
    reconstructed_data = np.dot(reduced_data,pca.components_)
    explained_variance = pca.explained_variance_ratio_
    error = mean_squared_error(data,reconstructed_data)
    print(f'Explained Variance Ratio: {sum(explained_variance)}, Reconstruction Error: {error}')

    # add labels back to reduced data
    reduced_data = np.concatenate((reduced_data,labels.reshape(-1,1)),axis=1)
    df = pd.DataFrame(reduced_data)
    # rename the target label based on dataset
    if dataset_name == 'csgo':
        column_name = num_dim
        df = df.rename(columns={column_name:'Round Winner'})
        plot_2d(df,f'{dataset_name}_pca_2d','Round Winner')
        plot_3d(df,f'{dataset_name}_pca_3d','Round Winner')
    else:
        column_name = num_dim
        df = df.rename(columns={column_name:'Exited'})
        plot_2d(df,f'{dataset_name}_pca_2d','Exited')
        plot_3d(df,f'{dataset_name}_pca_3d','Exited')

    kmeans_experiment(reduced_data,labels,dataset_name,dir='pca_data')

    gmm_experiment(reduced_data,labels,dataset_name, dir='pca_data')

    # evaluate the kmeans and gmm
    evaluate_kmeans(reduced_data,labels,15)

    evaluate_gmm(reduced_data,labels,14)
    

def pca_evaluate(data,labels,num_components,kmeans_clusters,gmm_components):
    pca = PCA(num_components)
    reduced_data = pca.fit_transform(data)

    # evaluate the kmeans and gmm
    print('KMEANS')
    evaluate_kmeans(reduced_data,labels,kmeans_clusters)
    print('GMM')
    evaluate_gmm(reduced_data,labels,gmm_components)


if __name__ == '__main__':

    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    # pca_reconstruction_metrics(cs_go_data,range(1,51),'csgo')

    # pca_reconstruction_metrics(loan_data,range(1,11),'loan')

    pca_experiment(cs_go_data,cs_go_labels,50,'csgo')

    pca_experiment(loan_data,loan_labels,6,'loan')

    pca_evaluate(cs_go_data,cs_go_labels,50,15,14)

    pca_evaluate(loan_data,loan_labels,6,9,11)