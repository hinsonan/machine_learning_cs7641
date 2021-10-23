from data_helpers import get_cs_go_data, get_loan_defualt
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from plot_helpers import plot_kurtosis, plot_reconstruction_error,plot_2d,plot_3d
from clustering import kmeans_experiment,gmm_experiment,evaluate_kmeans,evaluate_gmm
import numpy as np
import pandas as pd

def ica_reconstruction_metrics(data,range,dataset_name):
    errors = {}
    kurtosis_dic = {}
    for i in range:
        ica = FastICA(n_components=i)
        reduced_data = ica.fit_transform(data)

        # reconstruct the data and get the error
        reconstructed_data = ica.inverse_transform(reduced_data)
        ica_kurtosis = kurtosis(reconstructed_data)
        error = mean_squared_error(data,reconstructed_data)
        print(f'Range: {i}, Kurtosis {np.mean(ica_kurtosis)} Reconstruction Error: {error}')
        errors[i] = error
        kurtosis_dic[i] = np.mean(ica_kurtosis)
    plot_kurtosis(kurtosis_dic,f'{dataset_name}_kurtosis')
    plot_reconstruction_error(errors,f'{dataset_name}_reconstruction','ica_data')

def ica_experiment(data,labels,num_dim,dataset_name):
    ica = FastICA(n_components=num_dim)
    reduced_data = ica.fit_transform(data)

    # reconstruct the data and get the error
    reconstructed_data = ica.inverse_transform(reduced_data)
    ica_kurtosis = kurtosis(reconstructed_data)
    error = mean_squared_error(data,reconstructed_data)
    print(f'Kurtosis {np.mean(ica_kurtosis)} Reconstruction Error: {error}')

    # add labels back to reduced data
    reduced_data = np.concatenate((reduced_data,labels.reshape(-1,1)),axis=1)
    df = pd.DataFrame(reduced_data)
    # rename the target label based on dataset
    if dataset_name == 'csgo':
        column_name = num_dim
        df = df.rename(columns={column_name:'Round Winner'})
        plot_2d(df,f'{dataset_name}_ica_2d','Round Winner',dir='ica_data')
        plot_3d(df,f'{dataset_name}_ica_3d','Round Winner',dir='ica_data')
    else:
        column_name = num_dim
        df = df.rename(columns={column_name:'Exited'})
        plot_2d(df,f'{dataset_name}_ica_2d','Exited',dir='ica_data')
        plot_3d(df,f'{dataset_name}_ica_3d','Exited',dir='ica_data')

    kmeans_experiment(reduced_data,labels,dataset_name,dir='ica_data')

    gmm_experiment(reduced_data,labels,dataset_name, dir='ica_data')

def ica_evaluate(data,labels,num_components,kmeans_clusters,gmm_components):
    ica = FastICA(num_components)
    reduced_data = ica.fit_transform(data)

    # evaluate the kmeans and gmm
    print('KMEANS')
    evaluate_kmeans(reduced_data,labels,kmeans_clusters)
    print('GMM')
    evaluate_gmm(reduced_data,labels,gmm_components)

if __name__ == '__main__':
    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    # ica_reconstruction_metrics(cs_go_data,range(2,51),'csgo')

    # ica_reconstruction_metrics(loan_data,range(2,11),'loan')

    # ica_experiment(cs_go_data,cs_go_labels,40,'csgo')

    # ica_experiment(loan_data,loan_labels,8,'loan')

    ica_evaluate(cs_go_data,cs_go_labels,10,11,12)

    ica_evaluate(loan_data,loan_labels,8,9,11)
