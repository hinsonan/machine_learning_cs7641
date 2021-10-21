from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score
from data_helpers import get_cs_go_data, get_loan_defualt
from plot_helpers import plot_elbow, plot_silhouette, plot_homo_and_complete

def cs_go_clustering():
    print('Begin CSGO Clustering')
    clusters = list(range(2,41))
    sum_squared_distance = {}
    silhouette_scores = {}
    homo_scores = {}
    completeness_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        kmeans = KMeans(n_clusters=cluster, max_iter=500, random_state=0).fit(cs_go_data)
        sum_squared_distance[cluster] = kmeans.inertia_
        label = kmeans.labels_
        sil_score = silhouette_score(cs_go_data,label, metric='euclidean')
        homo_score = homogeneity_score(cs_go_labels,label)
        completeness = completeness_score(cs_go_labels,label)
        silhouette_scores[cluster] = sil_score
        homo_scores[cluster] = homo_score
        completeness_scores[cluster] = completeness
    plot_elbow(sum_squared_distance,'csgo_elbow')
    plot_silhouette(silhouette_scores,'csgo_silouette_km','KMeans Silhouette Score')
    plot_homo_and_complete(homo_scores,completeness_scores,'csgo_homo_and_complete_knn','KMeans Homogeneity and Completeness')


def loan_clustering():
    print('Begin Loan Clustering')
    clusters = list(range(2,41))
    sum_squared_distance = {}
    silhouette_scores = {}
    homo_scores = {}
    completeness_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        kmeans = KMeans(n_clusters=cluster, max_iter=500, random_state=0).fit(loan_data)
        sum_squared_distance[cluster] = kmeans.inertia_
        label = kmeans.labels_
        sil_score = silhouette_score(loan_data,label, metric='euclidean')
        homo_score = homogeneity_score(loan_labels,label)
        completeness = completeness_score(loan_labels,label)
        silhouette_scores[cluster] = sil_score
        homo_scores[cluster] = homo_score
        completeness_scores[cluster] = completeness
    plot_elbow(sum_squared_distance, 'loan_elbow')
    plot_silhouette(silhouette_scores,'loan_silouette_km','KMeans Silhouette Score')
    plot_homo_and_complete(homo_scores,completeness_scores,'loan_homo_and_complete_knn','KMeans Homogeneity and Completeness')

cs_go_data, cs_go_labels = get_cs_go_data()

loan_data, loan_labels  = get_loan_defualt()

cs_go_clustering()

loan_clustering()

