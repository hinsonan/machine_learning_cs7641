import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

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