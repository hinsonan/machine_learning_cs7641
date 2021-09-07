import pandas as pd
import numpy as np
import pickle, yaml, seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

def load_saved_model(name:str):
    with open(f'supervised_learning/models/{name}', 'rb') as f:
        clf = pickle.load(f)
    return clf

def one_hot_encode(df:pd.DataFrame, column:str, list_of_possible_strings_to_encode:list, is_bool=False):
    if is_bool:
        df[column].loc[(df[column] == True)] = 0
        df[column].loc[(df[column] == False)] = 1
        return
    for idx, string in enumerate(list_of_possible_strings_to_encode):
        df[column].loc[(df[column] == string)] = idx

def normalize_df(df:pd.DataFrame, exempt_cols:list=[]):
    for col in df.columns:
        if col in exempt_cols:
            continue
        df[col] = df[col] /df[col].abs().max()
    return df

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

def get_data():
    def get_cs_go_data():
        df = pd.read_csv('data/csgo_round_snapshots.csv')
        one_hot_encode(df,"round_winner",['CT','T'])
        one_hot_encode(df,"map",['de_dust2', 'de_mirage', 'de_nuke', 'de_inferno', 'de_overpass', 'de_vertigo', 'de_train', 'de_cache'])
        vals = normalize_with_min_max_scaler(df)
        data = vals[:,:-1]
        labels = vals[:,-1]
        return data, labels

    def get_breast_cancer_data():
        df = pd.read_csv('data/breast_cancer_data.csv')
        one_hot_encode(df,"diagnosis",['B','M'])
        vals = normalize_with_min_max_scaler(df)
        labels = vals[:,1].reshape(-1,1)
        data = np.delete(vals,1,1)
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

    
    with open('supervised_learning/dataset_config.yml','r') as f:
        config = yaml.load(f)
    
    if config['Active_Set'] == 'Dataset1':
        data, labels = get_cs_go_data()
    elif config['Active_Set'] == 'Dataset2':
        data, labels = get_breast_cancer_data()
    elif config['Active_Set'] == 'Dataset3':
        data, labels = get_loan_defualt()
    return data, labels

if __name__ == '__main__':
    df = pd.read_csv('data/csgo_round_snapshots.csv')
    one_hot_encode(df,"round_winner",['CT','T'])
    one_hot_encode(df,"map",['de_dust2', 'de_mirage', 'de_nuke', 'de_inferno', 'de_overpass', 'de_vertigo', 'de_train', 'de_cache'])
    seaborn.heatmap(df.corr())
    plt.show()