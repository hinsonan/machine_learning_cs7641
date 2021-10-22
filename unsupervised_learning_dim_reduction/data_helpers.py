import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from plot_helpers import pair_wise_plot

def normalize_with_min_max_scaler(df:pd.DataFrame):
    values = df.values
    scalar = MinMaxScaler()
    scaled_vals = scalar.fit_transform(values)
    return scaled_vals

def normalize_with_standard_scalar(df:pd.DataFrame, label_col:str):
    scalar = StandardScaler()
    scaled_df = pd.DataFrame(scalar.fit_transform(df), columns=df.columns)
    label_encoder = LabelEncoder()
    scaled_df[label_col] = label_encoder.fit_transform(scaled_df[label_col])
    return scaled_df.values


def get_cs_go_data():
    df = pd.read_csv('data/csgo_round_snapshots.csv')
    df['round_winner'] = LabelEncoder().fit_transform(df['round_winner'])
    df['map'] = LabelEncoder().fit_transform(df['map'])
    vals = normalize_with_min_max_scaler(df)
    data = vals[:,:-1]
    labels = vals[:,-1]
    return data, labels

def get_loan_defualt():
    df = pd.read_csv('data/loan_defaulter.csv')
    df = df.drop(['RowNumber','CustomerId','Surname'],axis=1)
    df['Geography'] = LabelEncoder().fit_transform(df['Geography'])
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    vals = normalize_with_min_max_scaler(df)
    data = vals[:,:-1]
    labels = vals[:,-1]
    return data, labels

if __name__ == '__main__':
    bank_loan = pd.read_csv('data/loan_defaulter.csv')
    bank_loan = bank_loan.drop(['RowNumber','CustomerId','Surname'],axis=1)
    bank_loan['Geography'] = LabelEncoder().fit_transform(bank_loan['Geography'])
    bank_loan['Gender'] = LabelEncoder().fit_transform(bank_loan['Gender'])

    cs_go = pd.read_csv('data/csgo_round_snapshots.csv')
    cs_go['round_winner'] = LabelEncoder().fit_transform(cs_go['round_winner'])
    cs_go['map'] = LabelEncoder().fit_transform(cs_go['map'])
    cs_go = cs_go.drop(labels=range(1000,27000),axis=0)

    pair_wise_plot(bank_loan,'Exited','bank_loan_pair_wise')
    pair_wise_plot(cs_go,'round_winner','csgo_pair_wise')