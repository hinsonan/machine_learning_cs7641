import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, accuracy_score, calinski_harabasz_score
from plot_helpers import plot_clusters, plot_elbow, plot_homo_and_complete, plot_silhouette, plot_calinski_harabasz, pair_wise_plot

def gmm_experiment(data,labels,dataset,dir):
    print('Begin GMM Clustering')
    clusters = list(range(2,41))
    silhouette_scores = {}
    calinski_harabasz_scores = {}
    homo_scores = {}
    completeness_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        gmm = GaussianMixture(n_components=cluster, max_iter=500, random_state=0).fit(data)
        label = gmm.predict(data)
        sil_score = silhouette_score(data,label, metric='euclidean')
        ch_score = calinski_harabasz_score(data,label)
        homo_score = homogeneity_score(labels,label)
        completeness = completeness_score(labels,label)
        silhouette_scores[cluster] = sil_score
        calinski_harabasz_scores[cluster] = ch_score
        homo_scores[cluster] = homo_score
        completeness_scores[cluster] = completeness
    plot_silhouette(silhouette_scores,f'{dataset}_silouette_gmm', 'GMM Silhouette Score', dir=dir)
    plot_calinski_harabasz(calinski_harabasz_scores,f'{dataset}_calinskiharabasz_gmm', 'GMM Calinski Harabasz', dir=dir)
    plot_homo_and_complete(homo_scores,completeness_scores,f'{dataset}_homo_and_complete_gmm','GMM Homogeneity and Completeness', dir=dir)

def kmeans_experiment(data,labels,dataset,dir):
    print('Begin KMeans Clustering')
    clusters = list(range(2,41))
    sum_squared_distance = {}
    silhouette_scores = {}
    calinski_harabasz_scores = {}
    homo_scores = {}
    completeness_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        kmeans = KMeans(n_clusters=cluster, max_iter=500, random_state=0).fit(data)
        sum_squared_distance[cluster] = kmeans.inertia_
        label = kmeans.labels_
        sil_score = silhouette_score(data,label, metric='euclidean')
        ch_score = calinski_harabasz_score(data,label)
        homo_score = homogeneity_score(labels,label)
        completeness = completeness_score(labels,label)
        silhouette_scores[cluster] = sil_score
        calinski_harabasz_scores[cluster] = ch_score
        homo_scores[cluster] = homo_score
        completeness_scores[cluster] = completeness
    plot_elbow(sum_squared_distance,f'{dataset}_elbow',dir=dir)
    plot_silhouette(silhouette_scores,f'{dataset}_silouette_kmeans','KMeans Silhouette Score', dir=dir)
    plot_calinski_harabasz(calinski_harabasz_scores,f'{dataset}_calinskiharabasz_kmeans', 'KMeans Calinski Harabasz', dir=dir)
    plot_homo_and_complete(homo_scores,completeness_scores,f'{dataset}_homo_and_complete_kmeans','KMeans Homogeneity and Completeness',dir=dir)

def evaluate_kmeans(data, truth_labels, num_clusters, dataset,dir):
    kmeans = KMeans(n_clusters=num_clusters,max_iter=500, random_state=0).fit(data)
    labels = kmeans.labels_
    plot_clusters(data,labels,'Kmeans',f'{dataset}_kmeans_clusters',dir=dir)
    # get kmeans metrics
    accuracy = accuracy_score(truth_labels,labels)
    sil_score = silhouette_score(data,labels, metric='euclidean')
    ch_score = calinski_harabasz_score(data,labels)
    homo_score = homogeneity_score(truth_labels,labels)
    completeness = completeness_score(truth_labels,labels)
    print(f'Accuracy: {accuracy}')
    print(f'Silhouette: {sil_score}')
    print(f'Calinski Harabasz: {ch_score}')
    print(f'Homogeneity: {homo_score}')
    print(f'Completeness: {completeness}')

    data = np.concatenate((data,np.array(labels).reshape(-1,1)),axis=1)
    # delete excess features that take forever to plot
    if data.shape[1] > 10:
        data = np.delete(data,list(range(10,data.shape[1]-1)),axis=1)
    data = pd.DataFrame(data)
    pair_wise_plot(data,data.columns[-1],f'{dataset}_kmeans_pairwise',dir=dir)

def evaluate_gmm(data, truth_labels, num_components,dataset,dir):
    gmm = GaussianMixture(n_components=num_components,max_iter=500, random_state=0).fit(data)
    labels = gmm.predict(data)

    plot_clusters(data,labels,'GMM',f'{dataset}_gmm_clusters',dir=dir)
    # get gmm metrics
    accuracy = accuracy_score(truth_labels,labels)
    sil_score = silhouette_score(data,labels, metric='euclidean')
    ch_score = calinski_harabasz_score(data,labels)
    homo_score = homogeneity_score(truth_labels,labels)
    completeness = completeness_score(truth_labels,labels)
    print(f'Accuracy: {accuracy}')
    print(f'Silhouette: {sil_score}')
    print(f'Calinski Harabasz: {ch_score}')
    print(f'Homogeneity: {homo_score}')
    print(f'Completeness: {completeness}')


    data = np.concatenate((data,np.array(labels).reshape(-1,1)),axis=1)
    # delete excess features that take forever to plot
    if data.shape[1] > 10:
        data = np.delete(data,list(range(10,data.shape[1]-1)),axis=1)
    data = pd.DataFrame(data)
    pair_wise_plot(data,data.columns[-1],f'{dataset}_gmm_pairwise',dir=dir)
    