import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_helper import one_hot_encode, normalize_with_min_max_scaler, normalize_with_standard_scalar
from neural_net import get_model
import os,pickle
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

def train_cs_go_nn(Xtrain,Xtest,Ytrain,Ytest):
    model = get_model(Xtrain.shape[-1])
    model.summary()
    model.fit(Xtrain,Ytrain, batch_size=128, epochs=2000, validation_data=(Xtest,Ytest))
    model.save('supervised_learning/models/neural_net_cs_go2')

def train_descision_tree_cs_go(Xtrain,Ytrain):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrain,Ytrain)
    with open('supervised_learning/models/decision_tree_cs_go', 'wb') as f:
        pickle.dump(clf, f)

def get_accuracy(model, data, truth):
    pred = model.predict(data)
    score = accuracy_score(truth.flatten(), pred.flatten())
    return score

df = pd.read_csv('data/csgo_round_snapshots.csv')
one_hot_encode(df,"round_winner",['CT','T'])
one_hot_encode(df,"map",['de_dust2', 'de_mirage', 'de_nuke', 'de_inferno', 'de_overpass', 'de_vertigo', 'de_train', 'de_cache'])
vals = normalize_with_standard_scalar(df, 'round_winner')
data = vals[:,:-1]
labels = vals[:,-1].reshape(-1,1)
# split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

# train_cs_go_nn(Xtrain, Xtest, Ytrain, Ytest)
# train_descision_tree_cs_go(Xtrain,Ytrain)

with open('supervised_learning/models/decision_tree_cs_go', 'rb') as f:
    clf = pickle.load(f)

score = get_accuracy(clf, Xtest, Ytest)
print(score)