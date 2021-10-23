from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, silhouette_score, homogeneity_score, completeness_score, accuracy_score
from data_helpers import get_cs_go_data, get_loan_defualt
from plot_helpers import plot_pca_2d, plot_pca_3d, plot_elbow, plot_homo_and_complete, plot_silhouette
from expectation_maximization import evaluate_gmm
from k_means_clustering import evaluate_kmeans
import numpy as np
import pandas as pd

def pca_reconstruction_metrics(data,range):
    for i in range:
        pca = PCA(n_components=i)
        reduced_data = pca.fit_transform(data)

        # reconstruct the data and get the error
        reconstructed_data = np.dot(reduced_data,pca.components_)
        explained_variance = pca.explained_variance_ratio_
        error = mean_squared_error(data,reconstructed_data)
        print(f'Range: {i} Explained Variance Ratio: {sum(explained_variance)}, Reconstruction Error: {error}')

def pca_kmeans(data,labels,dataset):
    print('Begin KMeans Clustering')
    clusters = list(range(2,41))
    sum_squared_distance = {}
    silhouette_scores = {}
    homo_scores = {}
    completeness_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        kmeans = KMeans(n_clusters=cluster, max_iter=500, random_state=0).fit(data)
        sum_squared_distance[cluster] = kmeans.inertia_
        label = kmeans.labels_
        sil_score = silhouette_score(data,label, metric='euclidean')
        homo_score = homogeneity_score(labels,label)
        completeness = completeness_score(labels,label)
        silhouette_scores[cluster] = sil_score
        homo_scores[cluster] = homo_score
        completeness_scores[cluster] = completeness
    plot_elbow(sum_squared_distance,f'{dataset}_elbow',dir='pca_data')
    plot_silhouette(silhouette_scores,f'{dataset}_silouette_km','KMeans Silhouette Score', dir='pca_data')
    plot_homo_and_complete(homo_scores,completeness_scores,f'{dataset}_homo_and_complete_knn','KMeans Homogeneity and Completeness',dir='pca_data')

def pca_gmm(data,labels,dataset):
    print('Begin GMM Clustering')
    clusters = list(range(2,41))
    silhouette_scores = {}
    homo_scores = {}
    completeness_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        gmm = GaussianMixture(n_components=cluster, max_iter=500, random_state=0).fit(data)
        label = gmm.predict(data)
        sil_score = silhouette_score(data,label, metric='euclidean')
        homo_score = homogeneity_score(labels,label)
        completeness = completeness_score(labels,label)
        silhouette_scores[cluster] = sil_score
        homo_scores[cluster] = homo_score
        completeness_scores[cluster] = completeness
    plot_silhouette(silhouette_scores,f'{dataset}_silouette_gmm', 'GMM Silhouette Score', dir='pca_data')
    plot_homo_and_complete(homo_scores,completeness_scores,f'{dataset}_homo_and_complete_gmm','GMM Homogeneity and Completeness', dir='pca_data')

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
    df = df.rename(columns={50:'Round Winner'})
    plot_pca_2d(df,f'{dataset_name}_pca_2d')
    plot_pca_3d(df,f'{dataset_name}_pca_3d')

    pca_kmeans(reduced_data,labels,dataset_name)

    pca_gmm(reduced_data,labels,dataset_name)

    # evaluate the kmeans and gmm
    evaluate_kmeans(reduced_data,labels,15)

    evaluate_gmm(reduced_data,labels,14)
    




if __name__ == '__main__':

    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    # pca_reconstruction_metrics(cs_go_data,range(1,51))

    # pca_reconstruction_metrics(loan_data,range(1,11))

    pca_experiment(cs_go_data,cs_go_labels,50,'csgo')

