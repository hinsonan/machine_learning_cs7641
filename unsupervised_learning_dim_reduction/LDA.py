from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from data_helpers import get_cs_go_data, get_loan_defualt
from sklearn.metrics import mean_squared_error,accuracy_score
from plot_helpers import plot_1d,plot_accuracy
from clustering import kmeans_experiment,gmm_experiment,evaluate_kmeans,evaluate_gmm
import numpy as np
import pandas as pd

def lda_experiment(data,labels,dataset_name):
    lda = LinearDiscriminantAnalysis()
    reduced_data = lda.fit_transform(data,labels)

    # get the accuracy for this set
    pred = lda.predict(data)
    acc_score = accuracy_score(labels,pred)
    print(f'LDA Accuracy: {acc_score}')

    # add labels back to reduced data
    reduced_data = np.concatenate((reduced_data,labels.reshape(-1,1)),axis=1)
    df = pd.DataFrame(reduced_data)
    # rename the target label based on dataset
    if dataset_name == 'csgo':
        column_name = 1
        df = df.rename(columns={column_name:'Round Winner'})
        plot_1d(df,f'{dataset_name}_lda_1d','Round Winner',dir='lda_data')
    else:
        column_name = 1
        df = df.rename(columns={column_name:'Exited'})
        plot_1d(df,f'{dataset_name}_lda_1d','Exited',dir='lda_data')

    kmeans_experiment(reduced_data,labels,dataset_name,dir='lda_data')

    gmm_experiment(reduced_data,labels,dataset_name, dir='lda_data')

def lda_evaluate(data,labels,dataset,kmeans_clusters,gmm_components,dir='lda_data'):
    lda = LinearDiscriminantAnalysis()
    reduced_data = lda.fit_transform(data,labels)

    # evaluate the kmeans and gmm
    print('KMEANS')
    evaluate_kmeans(reduced_data,labels,kmeans_clusters,dataset,dir)
    print('GMM')
    evaluate_gmm(reduced_data,labels,gmm_components,dataset,dir)

if __name__ == '__main__':
    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    # lda_experiment(cs_go_data,cs_go_labels,'csgo')

    # lda_experiment(loan_data,loan_labels,'loan')

    lda_evaluate(cs_go_data,cs_go_labels,'csgo',kmeans_clusters=35,gmm_components=35)

    lda_evaluate(loan_data,loan_labels,'loan',kmeans_clusters=15,gmm_components=25)
