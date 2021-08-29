import pandas as pd
from tensorflow.keras.models import load_model
from sklearn import tree, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_helper import one_hot_encode, normalize_with_min_max_scaler, normalize_with_standard_scalar
from neural_net import get_model
import os,pickle
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

def train_nn(Xtrain,Xtest,Ytrain,Ytest, model_name):
    model = get_model(Xtrain.shape[-1])
    model.summary()
    model.fit(Xtrain,Ytrain, batch_size=128, epochs=300, validation_data=(Xtest,Ytest))
    model.save(f'supervised_learning/models/{model_name}')

def train_descision_tree(Xtrain,Ytrain, model_name):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrain,Ytrain)
    with open(f'supervised_learning/models/{model_name}', 'wb') as f:
        pickle.dump(clf, f)

def train_SVM(Xtrain,Ytrain, model_name):
    clf = svm.SVC(kernel='rbf',verbose=True)
    clf.fit(Xtrain,Ytrain)
    with open(f'supervised_learning/models/{model_name}', 'wb') as f:
        pickle.dump(clf, f)

def train_booster(Xtrain, Ytrain, model_name):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(Xtrain,Ytrain)
    with open(f'supervised_learning/models/{model_name}', 'wb') as f:
        pickle.dump(clf, f)

def train_knn(Xtrain, Ytrain, model_name):
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(Xtrain,Ytrain)
    with open(f'supervised_learning/models/{model_name}', 'wb') as f:
        pickle.dump(clf, f)

def load_saved_model(name:str, is_nn=False):
    if is_nn:
        nn_model = load_model(f'supervised_learning/models/{name}')
        return nn_model
    with open(f'supervised_learning/models/{name}', 'rb') as f:
        clf = pickle.load(f)
    return clf

def get_accuracy(model, data, truth, is_prob=False):
    pred = model.predict(data)
    if is_prob:
        pred = np.where(pred<0.5,0,1)
    score = accuracy_score(truth.flatten(), pred.flatten())
    return score

def cs_go_training():
    df = pd.read_csv('data/csgo_round_snapshots.csv')
    one_hot_encode(df,"round_winner",['CT','T'])
    one_hot_encode(df,"map",['de_dust2', 'de_mirage', 'de_nuke', 'de_inferno', 'de_overpass', 'de_vertigo', 'de_train', 'de_cache'])
    vals = normalize_with_min_max_scaler(df)
    data = vals[:30000,:-1]
    labels = vals[:30000,-1].reshape(-1,1)
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    train_nn(Xtrain, Xtest, Ytrain, Ytest, 'neural_net_cs_go')
    # train_descision_tree(Xtrain,Ytrain, 'decision_tree_cs_go')
    # train_SVM(Xtrain,Ytrain, 'svm_cs_go')
    # train_booster(Xtrain,Ytrain,'booster_cs_go')
    # train_knn(Xtrain,Ytrain,'knn_cs_go')
    loaded_model = load_saved_model('neural_net_cs_go',True)
    score = get_accuracy(loaded_model, Xtest, Ytest,True)
    print(score)

def breast_cancer_training():
    df = pd.read_csv('data/breast_cancer_data.csv')
    one_hot_encode(df,"diagnosis",['B','M'])
    vals = normalize_with_min_max_scaler(df)
    labels = vals[:,1].reshape(-1,1)
    data = np.delete(vals,1,1)
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    # train_nn(Xtrain, Xtest, Ytrain, Ytest, 'neural_net_breast_cancer')
    # train_descision_tree(Xtrain,Ytrain, 'decision_tree_breast_cancer')
    # train_SVM(Xtrain,Ytrain, 'svm_breast_cancer')
    # train_booster(Xtrain,Ytrain,'booster_breast_cancer')
    # train_knn(Xtrain,Ytrain,'knn_breast_cancer')
    loaded_model = load_saved_model('knn_breast_cancer')
    score = get_accuracy(loaded_model, Xtest, Ytest)
    print(score)

cs_go_training()
#breast_cancer_training()