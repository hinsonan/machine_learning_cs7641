from boosting import AdaBoostClassifier, Booster
from neural_net import NeuralNet
from decision_tree import tree, DecisionTree
from knn import KNeighborsClassifier, KNN
from svm import svm, SVM
from data_helper import get_data
from sklearn.model_selection import train_test_split
import time, yaml

def wall_clock_experiment(Xtrain, Xtest, Ytrain, Ytest):
    model = NeuralNet.get_model(Xtrain.shape[-1])
    print('Begin Training NN')
    start = time.time()
    model.fit(Xtrain,Ytrain, batch_size=128, epochs=100, validation_data=(Xtest,Ytest), verbose=0)
    end = time.time()
    total_time_nn = end - start
    start = time.time()
    model.predict(Xtest)
    end = time.time()
    total_pred_time_nn = end - start

    print('Begin Training KNN')
    clf = KNeighborsClassifier(n_neighbors=1, weights="uniform")
    start = time.time()
    clf.fit(Xtrain,Ytrain)
    end = time.time()
    total_time_knn = end - start
    start = time.time()
    clf.predict(Xtest)
    end = time.time()
    total_pred_time_knn = end - start

    print('Begin Training SVM')
    clf = svm.SVC(kernel='poly',degree=5)
    start = time.time()
    clf.fit(Xtrain,Ytrain)
    end = time.time()
    total_time_svm = end - start
    start = time.time()
    clf.predict(Xtest)
    end = time.time()
    total_pred_time_svm = end - start

    print('Begin Training Booster')
    clf = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=tree.DecisionTreeClassifier(max_depth=30))
    start = time.time()
    clf.fit(Xtrain,Ytrain)
    end = time.time()
    total_time_booster = end - start
    start = time.time()
    clf.predict(Xtest)
    end = time.time()
    total_pred_time_boost = end - start

    print('Begin Training Decision Tree')
    clf = tree.DecisionTreeClassifier(max_depth=30, max_leaf_nodes=1100)
    start = time.time()
    clf = clf.fit(Xtrain,Ytrain)
    end = time.time()
    total_time_decision_tree = end - start
    start = time.time()
    clf.predict(Xtest)
    end = time.time()
    total_pred_time_tree = end - start

    names = ['Neural Network', 'KNN', 'SVM', 'AdaBoost', 'Decision Tree']
    values = [total_time_nn,total_time_knn,total_time_svm,total_time_booster,total_time_decision_tree]
    pred_times = [total_pred_time_nn,total_pred_time_knn,total_pred_time_svm,total_pred_time_boost, total_pred_time_tree]
    for name, val, pred in zip(names,values,pred_times):
        print(f'{name}: Training {val} seconds    Predict {pred} seconds')

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    wall_clock_experiment(Xtrain, Xtest, Ytrain, Ytest)
