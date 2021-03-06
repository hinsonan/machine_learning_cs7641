from scipy.sparse.sputils import matrix
from plot_helpers import plot_accuracy, plot_learning_curve, plot_multiple_learning_curves
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from data_helper import get_data, load_saved_model
import pickle, yaml
import numpy as np

class KNN:
    def __init__(self):
        pass

    def train_knn(self, Xtrain, Ytrain, model_name):
        clf = KNeighborsClassifier(n_neighbors=2)
        clf.fit(Xtrain,Ytrain)
        with open(f'supervised_learning/models/{model_name}', 'wb') as f:
            pickle.dump(clf, f)

    @staticmethod
    def get_accuracy(dataset_name, data, labels):
        model = load_saved_model(f'knn_{dataset_name}')
        pred = model.predict(data)
        score = accuracy_score(labels.flatten(), pred.flatten())
        print(f'KNN Accuracy: {score}')
        return score

    @staticmethod
    def get_precision_and_recall_scores(dataset_name, data, labels):
        model = load_saved_model(f'knn_{dataset_name}')
        pred = model.predict(data)
        p_score = precision_score(labels.flatten(), pred.flatten())
        r_score = recall_score(labels.flatten(), pred.flatten())
        print(f'KNN Precision: {p_score}')
        print(f'KNN Recall: {r_score}')
        return p_score, r_score

    def hyper_param_k(self, dataset_name, train_x, test_x, train_y, test_y):
        for i in [1,2,3,4,7]:
            clf = KNeighborsClassifier(n_neighbors=i)
            clf.fit(train_x,train_y)
            acc_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="accuracy")
            val_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="precision")
            rec_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="recall")
            print(f'KNN Accuracy with neighbor {i}: {np.mean(acc_scores)}')
            print(f'KNN Precision with neighbor {i}: {np.mean(val_scores)}')
            print(f'KNN Recall with neighbor {i}: {np.mean(rec_scores)}')
        
        # plot multiple learning curves
        estimators = [KNeighborsClassifier(n_neighbors=1),
                    KNeighborsClassifier(n_neighbors=2),
                    KNeighborsClassifier(n_neighbors=3),
                    KNeighborsClassifier(n_neighbors=6),
                    KNeighborsClassifier(n_neighbors=8)]
        plot_multiple_learning_curves(estimators, hyper_param_key='n_neighbors',title='KNN Learning Curve Using Different Neighbors',X=train_x, y=train_y, filename=f'{dataset_name}_knn_multi_learning_curve')

    def hyper_param_weights(self, dataset_name, train_x, test_x, train_y, test_y):
        for i in ["uniform", "distance"]:
            clf = KNeighborsClassifier(n_neighbors=2, weights=i)
            acc_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="accuracy")
            val_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="precision")
            rec_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="recall")
            print(f'KNN Accuracy with weights {i}: {np.mean(acc_scores)}')
            print(f'KNN Precision with weights {i}: {np.mean(val_scores)}')
            print(f'KNN Recall with weights {i}: {np.mean(rec_scores)}')
        
        # plot multiple learning curves
        estimators = [KNeighborsClassifier(n_neighbors=8, weights="uniform"),
                    KNeighborsClassifier(n_neighbors=8, weights="distance")]
        plot_multiple_learning_curves(estimators, hyper_param_key='weights',title='KNN Learning Curve Using Different Weights',X=train_x, y=train_y, filename=f'{dataset_name}_knn_multi_learning_curve_weights')

    def plot_learning_curve(self, data, labels, figname):
        plot_learning_curve(KNeighborsClassifier(n_neighbors=1, weights="uniform"),title="KNN Learning Curve",X=data,y=labels, filename=figname)

    def plot_test_set(self, dataset_name, train_x, test_x, train_y, test_y):
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_x,train_y)
        pred = clf.predict(test_x)
        score = accuracy_score(test_y.flatten(), pred.flatten())
        p_score = precision_score(test_y.flatten(), pred.flatten())
        r_score = recall_score(test_y.flatten(), pred.flatten())
        print(f'KNN Accuracy with neighbor: {score}')
        print(f'KNN Precision with neighbor: {p_score}')
        print(f'KNN Recall with neighbor: {r_score}')
        plot_accuracy(np.sort(test_y.flatten()), np.sort(pred.flatten()), f'{dataset_name}_knn_testing_set')

    def gather_metrics(self, train_x, test_x, train_y, test_y):
        clf = KNeighborsClassifier(n_neighbors=1, weights="uniform")
        clf.fit(train_x,train_y)
        pred = clf.predict(test_x)
        score = accuracy_score(test_y.flatten(), pred.flatten())
        p_score = precision_score(test_y.flatten(), pred.flatten())
        r_score = recall_score(test_y.flatten(), pred.flatten())
        print(f'KNN Accuracy: {score}')
        print(f'KNN Precision: {p_score}')
        print(f'KNN Recall: {r_score}')

    def run_all(self,dataset, Xtrain, Xtest, Ytrain, Ytest):
        self.plot_learning_curve(Xtrain, Ytrain, f'{dataset}_booster_learning_curve')
        self.hyper_param_k(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.hyper_param_weights(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42, shuffle=True)

    knn = KNN()

    #knn.train_knn(Xtrain, Ytrain, f'knn_{DATASET_NAME}')

    # knn.get_accuracy(DATASET_NAME, Xtest, Ytest)

    # knn.get_precision_and_recall_scores(DATASET_NAME, Xtest, Ytest)

    knn.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)

    knn.plot_learning_curve(Xtrain,Ytrain,f'{DATASET_NAME}_knn_learning_curve')

    # knn.hyper_param_k(DATASET_NAME,Xtrain, Xtest, Ytrain, Ytest)

    # knn.hyper_param_weights(DATASET_NAME,Xtrain, Xtest, Ytrain, Ytest)

    # knn.plot_test_set(DATASET_NAME, Xtrain, Xtest, Ytrain, Ytest)