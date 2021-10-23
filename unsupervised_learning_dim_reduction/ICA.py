from data_helpers import get_cs_go_data, get_loan_defualt
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import FastICA

def ica_reconstruction_metrics(data,range):
    for i in range:
        ica = FastICA(n_components=i)
        reduced_data = ica.fit_transform(data)

        # reconstruct the data and get the error
        reconstructed_data = ica.inverse_transform(reduced_data)
        error = mean_squared_error(data,reconstructed_data)
        print(f'Range: {i},  Reconstruction Error: {error}')


if __name__ == '__main__':
    cs_go_data, cs_go_labels = get_cs_go_data()

    loan_data, loan_labels  = get_loan_defualt()

    ica_reconstruction_metrics(cs_go_data,range(2,50))

    ica_reconstruction_metrics(loan_data,range(2,10))