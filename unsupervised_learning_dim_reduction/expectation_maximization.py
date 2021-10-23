from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, accuracy_score
from data_helpers import get_cs_go_data, get_loan_defualt
from plot_helpers import plot_silhouette, plot_homo_and_complete

def cs_go_gmm():
    print('Begin CSGO Clustering')
    clusters = list(range(2,41))
    silhouette_scores = {}
    homo_scores = {}
    completeness_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        gmm = GaussianMixture(n_components=cluster, max_iter=500, random_state=0).fit(cs_go_data)
        label = gmm.predict(cs_go_data)
        sil_score = silhouette_score(cs_go_data,label, metric='euclidean')
        homo_score = homogeneity_score(cs_go_labels,label)
        completeness = completeness_score(cs_go_labels,label)
        silhouette_scores[cluster] = sil_score
        homo_scores[cluster] = homo_score
        completeness_scores[cluster] = completeness
    plot_silhouette(silhouette_scores,'csgo_silouette_gmm', 'GMM Silhouette Score')
    plot_homo_and_complete(homo_scores,completeness_scores,'csgo_homo_and_complete_gmm','GMM Homogeneity and Completeness')


def loan_gmm():
    print('Begin Loan Clustering')
    clusters = list(range(2,41))
    silhouette_scores = {}
    homo_scores = {}
    completeness_scores = {}
    for idx,cluster in enumerate(clusters):
        print(f'On Iteration {idx}')
        gmm = GaussianMixture(n_components=cluster, max_iter=500, random_state=0).fit(loan_data)
        label = gmm.predict(loan_data)
        sil_score = silhouette_score(loan_data,label, metric='euclidean')
        homo_score = homogeneity_score(loan_labels,label)
        completeness = completeness_score(loan_labels,label)
        silhouette_scores[cluster] = sil_score
        homo_scores[cluster] = homo_score
        completeness_scores[cluster] = completeness
    plot_silhouette(silhouette_scores,'loan_silouette_gmm', 'GMM Silhouette Score')
    plot_homo_and_complete(homo_scores,completeness_scores,'loan_homo_and_complete_gmm','GMM Homogeneity and Completeness')

def evaluate_gmm(data, truth_labels, num_components):
    gmm = GaussianMixture(n_components=num_components,max_iter=500, random_state=0).fit(data)
    labels = gmm.predict(data)
    accuracy = accuracy_score(truth_labels,labels)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    cs_go_gmm()

    loan_gmm()

    evaluate_gmm(cs_go_data,cs_go_labels,5)

    evaluate_gmm(loan_data,loan_labels,5)