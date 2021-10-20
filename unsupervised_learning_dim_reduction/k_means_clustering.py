from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_helpers import get_cs_go_data, get_loan_defualt
from plot_helpers import plot_elbow, plot_silhouette

def cs_go_clustering():
    print('Begin CSGO Clustering')
    clusters = list(range(2,41))
    sum_squared_distance = {}
    silhouette_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        kmeans = KMeans(n_clusters=cluster, max_iter=500, random_state=0).fit(cs_go_data)
        sum_squared_distance[cluster] = kmeans.inertia_
        label = kmeans.labels_
        sil_score = silhouette_score(cs_go_data,label, metric='euclidean')
        silhouette_scores[cluster] = sil_score
    plot_elbow(sum_squared_distance,'csgo_elbow')
    plot_silhouette(silhouette_scores,'csgo_silouette')


def loan_clustering():
    print('Begin Loan Clustering')
    clusters = list(range(2,41))
    sum_squared_distance = {}
    silhouette_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        kmeans = KMeans(n_clusters=cluster, max_iter=500, random_state=0).fit(loan_data)
        sum_squared_distance[cluster] = kmeans.inertia_
        label = kmeans.labels_
        sil_score = silhouette_score(loan_data,label, metric='euclidean')
        silhouette_scores[cluster] = sil_score
    plot_elbow(sum_squared_distance, 'loan_elbow')
    plot_silhouette(silhouette_scores,'loan_silouette')

cs_go_data, cs_go_labels = get_cs_go_data()

loan_data, loan_labels  = get_loan_defualt()

cs_go_clustering()

loan_clustering()

