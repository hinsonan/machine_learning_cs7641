from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from data_helpers import get_cs_go_data, get_loan_defualt
import numpy as np

def pca_reconstruction_metrics(data,range):
    for i in range:
        pca = PCA(n_components=i)
        reduced_data = pca.fit_transform(data)

        # reconstruct the data and get the error
        reconstructed_data = np.dot(reduced_data,pca.components_)
        explained_variance = pca.explained_variance_ratio_
        error = mean_squared_error(data,reconstructed_data)
        print(f'Range: {i} Explained Variance Ratio: {sum(explained_variance)}, Reconstruction Error: {error}')

def pca_experiment(data,num_dim):
    pca = PCA(n_components=num_dim)
    reduced_data = pca.fit_transform(data)

    # reconstruct the data and get the error
    reconstructed_data = np.dot(reduced_data,pca.components_)
    explained_variance = pca.explained_variance_ratio_
    error = mean_squared_error(data,reconstructed_data)





if __name__ == '__main__':

    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    # pca_reconstruction_metrics(cs_go_data,range(1,51))

    # pca_reconstruction_metrics(loan_data,range(1,11))

    pca_experiment(cs_go_data,50)



