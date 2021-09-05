from boosting import AdaBoostClassifier, accuracy_booster
from neural_net import get_model, accuracy_nn
from decision_tree import tree, accuracy_dt
from knn import KNeighborsClassifier, accuracy_knn
from svm import svm, accuracy_svm
from data_helper import get_data
from sklearn.model_selection import train_test_split
import time, yaml

def wall_clock_experiment(Xtrain, Xtest, Ytrain, Ytest):
    model = get_model(Xtrain.shape[-1])
    print('Begin Training NN')
    start = time.time()
    model.fit(Xtrain,Ytrain, batch_size=128, epochs=300, validation_data=(Xtest,Ytest))
    end = time.time()
    total_time_nn = end - start

    print('Begin Training KNN')
    clf = KNeighborsClassifier(n_neighbors=2)
    start = time.time()
    clf.fit(Xtrain,Ytrain)
    end = time.time()
    total_time_knn = end - start

    print('Begin Training SVM')
    clf = svm.SVC(kernel='rbf',verbose=True)
    start = time.time()
    clf.fit(Xtrain,Ytrain)
    end = time.time()
    total_time_svm = end - start

    print('Begin Training Booster')
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    start = time.time()
    clf.fit(Xtrain,Ytrain)
    end = time.time()
    total_time_booster = end - start

    print('Begin Training Decision Tree')
    clf = tree.DecisionTreeClassifier()
    start = time.time()
    clf = clf.fit(Xtrain,Ytrain)
    end = time.time()
    total_time_decision_tree = end - start

    names = ['Neural Network', 'KNN', 'SVM', 'AdaBoost', 'Decision Tree']
    values = [total_time_nn,total_time_knn,total_time_svm,total_time_booster,total_time_decision_tree]
    for name, val in zip(names,values):
        print(f'{name}: {val} seconds')

def accuracy_comparison(dataset_name, data, labels):
    accuracy_nn(dataset_name, data, labels)
    accuracy_booster(dataset_name, data, labels)
    accuracy_dt(dataset_name, data, labels)
    accuracy_knn(dataset_name, data, labels)
    accuracy_svm(dataset_name, data, labels)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    #wall_clock_experiment(Xtrain, Xtest, Ytrain, Ytest)

    accuracy_comparison(DATASET_NAME, Xtest, Ytest)