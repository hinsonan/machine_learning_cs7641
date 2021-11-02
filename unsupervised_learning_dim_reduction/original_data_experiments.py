from data_helpers import get_cs_go_data, get_loan_defualt
from clustering import kmeans_experiment, gmm_experiment, evaluate_kmeans, evaluate_gmm

if __name__ == '__main__':
    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    kmeans_experiment(cs_go_data,cs_go_labels,'csgo',dir='original_data')

    gmm_experiment(cs_go_data,cs_go_labels,'csgo',dir='original_data')

    kmeans_experiment(loan_data,loan_labels,'loan',dir='original_data')

    gmm_experiment(loan_data,loan_labels,'loan',dir='original_data')

    evaluate_kmeans(cs_go_data,cs_go_labels,4,'csgo','original_data')

    evaluate_gmm(cs_go_data,cs_go_labels,4,'csgo','original_data')

    evaluate_kmeans(loan_data,loan_labels,9,'loan','original_data')

    evaluate_gmm(loan_data,loan_labels,11,'loan','original_data')


