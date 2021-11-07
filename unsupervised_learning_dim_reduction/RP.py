from sklearn.random_projection import GaussianRandomProjection
from data_helpers import get_cs_go_data, get_loan_defualt
from sklearn.metrics import mean_squared_error
from plot_helpers import plot_reconstruction_error,plot_2d,plot_3d
from clustering import kmeans_experiment,gmm_experiment,evaluate_kmeans,evaluate_gmm
import numpy as np
import pandas as pd

def rp_reconstruction_metrics(data,range,dataset_name):
    errors = {}
    for i in range:
        rp = GaussianRandomProjection(n_components=i, random_state=0)
        reduced_data = rp.fit_transform(data)

        # reconstruct the data and get the error
        reconstructed_data = np.dot(reduced_data,rp.components_)
        error = mean_squared_error(data,reconstructed_data)
        print(f'Range: {i}, Reconstruction Error: {error}')
        errors[i] = error
    plot_reconstruction_error(errors,f'{dataset_name}_reconstruction','rp_data')

def rp_experiment(data,labels,num_dim,dataset_name):
    rp = GaussianRandomProjection(n_components=num_dim, random_state=0)
    reduced_data = rp.fit_transform(data)

    # reconstruct the data and get the error
    reconstructed_data = np.dot(reduced_data,rp.components_)
    error = mean_squared_error(data,reconstructed_data)
    print(f'Reconstruction Error: {error}')

    # add labels back to reduced data
    reduced_data = np.concatenate((reduced_data,labels.reshape(-1,1)),axis=1)
    df = pd.DataFrame(reduced_data)
    # rename the target label based on dataset
    if dataset_name == 'csgo':
        column_name = num_dim
        df = df.rename(columns={column_name:'Round Winner'})
        plot_2d(df,f'{dataset_name}_rp_2d','Round Winner',dir='rp_data')
        plot_3d(df,f'{dataset_name}_rp_3d','Round Winner',dir='rp_data')
    else:
        column_name = num_dim
        df = df.rename(columns={column_name:'Exited'})
        plot_2d(df,f'{dataset_name}_rp_2d','Exited',dir='rp_data')
        plot_3d(df,f'{dataset_name}_rp_3d','Exited',dir='rp_data')

    kmeans_experiment(reduced_data,labels,dataset_name,dir='rp_data')

    gmm_experiment(reduced_data,labels,dataset_name, dir='rp_data')

def rp_evaluate(data,labels,dataset,dr_components,kmeans_clusters,gmm_components,dir='rp_data'):
    rp = GaussianRandomProjection(dr_components, random_state=0)
    reduced_data = rp.fit_transform(data)

    # evaluate the kmeans and gmm
    print('KMEANS')
    evaluate_kmeans(reduced_data,labels,kmeans_clusters,dataset,dir)
    print('GMM')
    evaluate_gmm(reduced_data,labels,gmm_components,dataset,dir)

if __name__ == '__main__':
    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    rp_reconstruction_metrics(cs_go_data,range(2,89),'csgo')

    rp_reconstruction_metrics(loan_data,range(2,11),'loan')

    rp_experiment(cs_go_data,cs_go_labels,70,'csgo')

    rp_experiment(loan_data,loan_labels,10,'loan')

    rp_evaluate(cs_go_data,cs_go_labels,'csgo',dr_components=70,kmeans_clusters=4,gmm_components=7)

    rp_evaluate(loan_data,loan_labels,'loan',dr_components=10,kmeans_clusters=9,gmm_components=11)
