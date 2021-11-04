from data_helpers import get_cs_go_data, get_loan_defualt
from clustering import kmeans_experiment, gmm_experiment, evaluate_kmeans, evaluate_gmm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

def plot_orig_data(df,fig_name,target_name,dir='pca_data'):
    fig = plt.figure()
    ax = Axes3D(fig)

    sequence_containing_x_vals = df[0]
    sequence_containing_y_vals = df[1]
    sequence_containing_z_vals = df[2]

    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals,c=df[target_name])
    ax.set_title("Data Reduction")
    ax.set_xlabel('Reduction Feature 1')
    ax.set_ylabel('Reduction Feature 2')
    ax.set_zlabel('Reduction Feature 3')
    plt.savefig(f'unsupervised_learning_dim_reduction/charts/{dir}/{fig_name}')
    plt.clf()

if __name__ == '__main__':
    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    # plot_orig_data(pd.DataFrame(np.concatenate((cs_go_data,cs_go_labels.reshape(-1,1)),axis=1)),'csgo_orginal_3d',96,dir='original_data')

    # plot_orig_data(pd.DataFrame(np.concatenate((loan_data,loan_labels.reshape(-1,1)),axis=1)),'loan_orginal_3d',10,dir='original_data')

    kmeans_experiment(cs_go_data,cs_go_labels,'csgo',dir='original_data')

    gmm_experiment(cs_go_data,cs_go_labels,'csgo',dir='original_data')

    kmeans_experiment(loan_data,loan_labels,'loan',dir='original_data')

    gmm_experiment(loan_data,loan_labels,'loan',dir='original_data')

    evaluate_kmeans(cs_go_data,cs_go_labels,4,'csgo','original_data')

    evaluate_gmm(cs_go_data,cs_go_labels,4,'csgo','original_data')

    evaluate_kmeans(loan_data,loan_labels,9,'loan','original_data')

    evaluate_gmm(loan_data,loan_labels,11,'loan','original_data')


