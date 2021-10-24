from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, accuracy_score
from plot_helpers import plot_elbow, plot_homo_and_complete, plot_silhouette

def gmm_experiment(data,labels,dataset,dir):
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
    plot_silhouette(silhouette_scores,f'{dataset}_silouette_gmm', 'GMM Silhouette Score', dir=dir)
    plot_homo_and_complete(homo_scores,completeness_scores,f'{dataset}_homo_and_complete_gmm','GMM Homogeneity and Completeness', dir=dir)

def kmeans_experiment(data,labels,dataset,dir):
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
    plot_elbow(sum_squared_distance,f'{dataset}_elbow',dir=dir)
    plot_silhouette(silhouette_scores,f'{dataset}_silouette_km','KMeans Silhouette Score', dir=dir)
    plot_homo_and_complete(homo_scores,completeness_scores,f'{dataset}_homo_and_complete_knn','KMeans Homogeneity and Completeness',dir=dir)

def evaluate_kmeans(data, truth_labels, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters,max_iter=500, random_state=0).fit(data)
    labels = kmeans.labels_
    accuracy = accuracy_score(truth_labels,labels)
    print(f'Accuracy: {accuracy}')

def evaluate_gmm(data, truth_labels, num_components):
    gmm = GaussianMixture(n_components=num_components,max_iter=500, random_state=0).fit(data)
    labels = gmm.predict(data)
    accuracy = accuracy_score(truth_labels,labels)
    print(f'Accuracy: {accuracy}')