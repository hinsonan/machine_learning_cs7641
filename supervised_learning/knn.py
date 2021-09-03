from plot_helpers import plot_learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_helper import get_breast_cancer_data, get_cs_go_data, load_saved_model
import pickle

def train_knn(Xtrain, Ytrain, model_name):
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(Xtrain,Ytrain)
    with open(f'supervised_learning/models/{model_name}', 'wb') as f:
        pickle.dump(clf, f)

data, labels = get_cs_go_data()
# split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

#train_knn(Xtrain, Ytrain, 'knn_cs_go')

model = load_saved_model('knn_cs_go')
pred = model.predict(Xtest)
score = accuracy_score(Ytest.flatten(), pred.flatten())
print(score)
plot_learning_curve(KNeighborsClassifier(n_neighbors=2),'KNN',Xtrain[:30000],Ytrain[:30000])