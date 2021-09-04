from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_helper import get_breast_cancer_data, get_cs_go_data, load_saved_model
import pickle

def train_booster(Xtrain, Ytrain, model_name):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(Xtrain,Ytrain)
    with open(f'supervised_learning/models/{model_name}', 'wb') as f:
        pickle.dump(clf, f)

data, labels = get_cs_go_data()
# split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

#train_booster(Xtrain, Ytrain, 'booster_cs_go')

model = load_saved_model('booster_cs_go')
pred = model.predict(Xtest)
score = accuracy_score(Ytest.flatten(), pred.flatten())
print(score)