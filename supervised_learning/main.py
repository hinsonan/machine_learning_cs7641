from math import degrees
from data_helper import get_data
from numpy.testing._private.utils import KnownFailureException
from sklearn import neighbors
from boosting import AdaBoostClassifier, Booster
from neural_net import NeuralNet
from decision_tree import tree, DecisionTree
from knn import KNeighborsClassifier, KNN
from svm import svm, SVM
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from combined_experiments import wall_clock_experiment
from plot_helpers import plot_learning_curve, plot_neural_net_history_accuracy, plot_neural_net_history_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import yaml,os
def get_nn(dataset_num):
    if dataset_num == 1:
        inputs = Input(shape=(96))
        layer = Dense(312, activation='relu')(inputs)
        layer = Dense(212, activation='relu')(inputs)
        layer = Dense(128, activation='relu')(layer)
        layer = Dense(64, activation='relu')(layer)
        layer = Dense(32, activation='relu')(layer)
        layer = Dense(16, activation='relu')(layer)
        layer = Dense(8, activation='relu')(layer)
        layer = Dense(4, activation='relu')(layer)
        output = Dense(1, activation="sigmoid")(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        return model
    else:
        inputs = Input(shape=(10))
        layer = Dense(8, activation="swish")(inputs)
        layer = Dense(4, activation="swish")(layer)
        output = Dense(1, activation="sigmoid")(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        return model

def gather_metrics(clf, name, train_x, test_x, train_y, test_y):
        acc_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="accuracy")
        val_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="precision")
        rec_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="recall")
        print("******************************VALIDATION SCORES*********************************")
        print(f'{name} Accuracy with max depth: {np.mean(acc_scores)}')
        print(f'{name} Precision with max depth: {np.mean(val_scores)}')
        print(f'{name} Recall with max depth: {np.mean(rec_scores)}')
        clf.fit(train_x,train_y)
        pred = clf.predict(test_x)
        score = accuracy_score(test_y.flatten(), pred.flatten())
        p_score = precision_score(test_y.flatten(), pred.flatten())
        r_score = recall_score(test_y.flatten(), pred.flatten())
        print("******************************TEST SCORES*********************************")
        print(f'{name} Accuracy: {score}')
        print(f'{name} Precision: {p_score}')
        print(f'{name} Recall: {r_score}')

def plot_learning_curve_nn(model, name, dataset_name, train_x, train_y, usedefault=False):
        if usedefault:
            inputs = Input(shape=(train_x.shape[-1]))
            layer = Dense(32, activation='relu')(inputs)
            layer = Dense(8, activation='relu')(layer)
            layer = Dense(4, activation='relu')(layer)
            output = Dense(1, activation='sigmoid')(layer)
            model = Model(inputs, output)
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])


        Xtrain, Xval, Ytrain, Yval = train_test_split(train_x,train_y, test_size=0.25, random_state=42, shuffle=True)
        history = model.fit(Xtrain,Ytrain, batch_size=128, epochs=100, validation_data=(Xval,Yval),verbose=0)
        plot_neural_net_history_accuracy(history.history,f'{dataset_name}_nn_{name}_accuracy')
        plot_neural_net_history_loss(history.history, f'{dataset_name}_nn_{name}_loss')

def gather_nn_metrics(model, train_x, test_x, train_y, test_y):
        train_x, Xval, train_y, Yval = train_test_split(train_x,train_y, test_size=0.25, random_state=42, shuffle=True)
        model.fit(train_x, train_y, batch_size=128, epochs=100, verbose=0)
        pred = model.predict(Xval)
        pred = np.where(pred<0.5,0,1)
        score = accuracy_score(Yval.flatten(), pred.flatten())
        p_score = precision_score(Yval.flatten(), pred.flatten())
        r_score = recall_score(Yval.flatten(), pred.flatten())
        print('******************************VALIDATION SCORES*********************************')
        print(f'Nueral Net Accuracy: {score}')
        print(f'Neural Net Precision: {p_score}')
        print(f'Neural Net Recall: {r_score}')

        model.fit(train_x, train_y, batch_size=128, epochs=100, verbose=0)
        pred = model.predict(test_x)
        pred = np.where(pred<0.5,0,1)
        score = accuracy_score(test_y.flatten(), pred.flatten())
        p_score = precision_score(test_y.flatten(), pred.flatten())
        r_score = recall_score(test_y.flatten(), pred.flatten())
        print('******************************TEST SCORES*********************************')
        print(f'Nueral Net Accuracy: {score}')
        print(f'Neural Net Precision: {p_score}')
        print(f'Neural Net Recall: {r_score}')

with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

DATASET_NAME = config[config['Active_Set']]['name']

if DATASET_NAME == 'cs_go':
    # Dataset 1 tuned models
    knn_tuned = KNeighborsClassifier(n_neighbors=1)
    svm_tuned = svm.SVC(kernel="poly", degree=5)
    dt_tuned = tree.DecisionTreeClassifier(max_depth=30, max_leaf_nodes=1100)
    booster_tuned = AdaBoostClassifier(n_estimators=100, base_estimator=tree.DecisionTreeClassifier(max_depth=30))
    nn_tuned = get_nn(1)
else:
    # Dataset 2 tuned models
    knn_tuned = KNeighborsClassifier(n_neighbors=6, weights='uniform')
    svm_tuned = svm.SVC(kernel="poly", degree=2)
    dt_tuned = tree.DecisionTreeClassifier(max_depth=30, max_leaf_nodes=1100)
    booster_tuned = AdaBoostClassifier(n_estimators=5, base_estimator=tree.DecisionTreeClassifier(max_depth=3))
    nn_tuned = get_nn(2)

data, labels = get_data()
# split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42, shuffle=True)

plot_learning_curve(KNeighborsClassifier(), title="KNN Learning Curve",X=Xtrain,y=Ytrain,filename=f'{DATASET_NAME}_knn_initial_learning_curve')
plot_learning_curve(svm.SVC(), title="SVM Learning Curve",X=Xtrain,y=Ytrain,filename=f'{DATASET_NAME}_svm_initial_learning_curve')
plot_learning_curve(tree.DecisionTreeClassifier(), title="Decision Tree Learning Curve",X=Xtrain,y=Ytrain,filename=f'{DATASET_NAME}_dt_initial_learning_curve')
plot_learning_curve(AdaBoostClassifier(), title="Boosting Learning Curve",X=Xtrain,y=Ytrain,filename=f'{DATASET_NAME}_boost_initial_learning_curve')
plot_learning_curve_nn(nn_tuned, "initial", DATASET_NAME, Xtrain, Ytrain, True)

gather_metrics(knn_tuned, 'knn', Xtrain, Xtest, Ytrain, Ytest)
gather_metrics(svm_tuned, 'svm', Xtrain, Xtest, Ytrain, Ytest)
gather_metrics(dt_tuned, 'dt', Xtrain, Xtest, Ytrain, Ytest)
gather_metrics(booster_tuned, 'booster', Xtrain, Xtest, Ytrain, Ytest)
gather_nn_metrics(nn_tuned, Xtrain, Xtest, Ytrain, Ytest)

plot_learning_curve(knn_tuned, title="KNN Learning Curve",X=Xtrain,y=Ytrain,filename=f'{DATASET_NAME}_knn_final_learning_curve')
plot_learning_curve(svm_tuned, title="SVM Learning Curve",X=Xtrain,y=Ytrain,filename=f'{DATASET_NAME}_svm_final_learning_curve')
plot_learning_curve(dt_tuned, title="Decision Tree Learning Curve",X=Xtrain,y=Ytrain,filename=f'{DATASET_NAME}_dt_final_learning_curve')
plot_learning_curve(booster_tuned, title="Boosting Learning Curve",X=Xtrain,y=Ytrain,filename=f'{DATASET_NAME}_boost_final_learning_curve')
plot_learning_curve_nn(nn_tuned, "final", DATASET_NAME, Xtrain, Ytrain)

wall_clock_experiment(Xtrain, Xtest, Ytrain, Ytest)


