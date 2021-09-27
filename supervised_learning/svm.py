from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, hinge_loss
from sklearn.model_selection import train_test_split, cross_val_score
from data_helper import get_data, load_saved_model
from plot_helpers import plot_learning_curve, plot_multiple_learning_curves, plot_svm_iterative_learning_curve
import pickle, yaml
import numpy as np

class SVM():

    def __init__(self) -> None:
        pass

    def train_SVM(self, Xtrain,Ytrain, model_name):
        clf = svm.SVC(kernel='rbf',verbose=True)
        clf.fit(Xtrain,Ytrain)
        with open(f'supervised_learning/models/{model_name}', 'wb') as f:
            pickle.dump(clf, f)

    @staticmethod
    def get_accuracy(dataset_name, data, labels):
        model = load_saved_model(f'svm_{dataset_name}')
        pred = model.predict(data)
        score = accuracy_score(labels.flatten(), pred.flatten())
        print(f'SVM Accuracy: {score}')
        return score

    @staticmethod
    def get_precision_and_recall_scores(dataset_name, data, labels):
        model = load_saved_model(f'svm_{dataset_name}')
        pred = model.predict(data)
        p_score = precision_score(labels.flatten(), pred.flatten())
        r_score = recall_score(labels.flatten(), pred.flatten())
        print(f'SVM Precision: {p_score}')
        print(f'SVM Recall: {r_score}')
        return p_score, r_score

    def hyper_param_kernel(self, dataset_name, train_x, test_x, train_y, test_y):
        for i in ['rbf','poly','linear','sigmoid']:
            clf = svm.SVC(kernel=i)
            clf.fit(train_x,train_y)
            acc_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="accuracy")
            val_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="precision")
            rec_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="recall")
            print(f'SVM Accuracy with kernel {i}: {np.mean(acc_scores)}')
            print(f'SVM Precision with kernel {i}: {np.mean(val_scores)}')
            print(f'SVM Recall with kernel {i}: {np.mean(rec_scores)}')
        
        # plot multiple learning curves
        estimators = [svm.SVC(kernel='rbf'),
                    svm.SVC(kernel='poly'),
                    svm.SVC(kernel='linear'),
                    svm.SVC(kernel='sigmoid')]
        plot_multiple_learning_curves(estimators, hyper_param_key='kernel',title='SVM performance using different kernels',X=train_x, y=train_y, filename=f'{dataset_name}_svm_multi_learning_curve')

    def hyper_param_degree(self, dataset_name, train_x, test_x, train_y, test_y):
        for i in [5,8]:
            clf = svm.SVC(kernel="poly", degree=i)
            clf.fit(train_x,train_y)
            acc_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="accuracy")
            val_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="precision")
            rec_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="recall")
            print(f'SVM Accuracy with kernel {i}: {np.mean(acc_scores)}')
            print(f'SVM Precision with kernel {i}: {np.mean(val_scores)}')
            print(f'SVM Recall with kernel {i}: {np.mean(rec_scores)}')
            print(f'*****end of {i}********')

    def plot_learning_curve(self, data, labels, figname):
        plot_learning_curve(svm.SVC(kernel="poly", degree=2),title="SVM Learning Curve",X=data,y=labels,filename=figname)

    def plot_iterative_curve(self, dataset_name, data, labels):
        scores = []
        for i in range(1,400):
            clf = svm.SVC(kernel="poly", degree=2, max_iter=i)
            clf.fit(data,labels)
            pred = clf.predict(data)
            score = hinge_loss(labels.flatten(), pred.flatten())
            scores.append(score)
        plot_svm_iterative_learning_curve(scores, f'{dataset_name}_svm_iterative_learning_curve')
        
    def gather_metrics(self, train_x, test_x, train_y, test_y):
        clf = svm.SVC(kernel="poly", degree=2)
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
        self.hyper_param_kernel(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.hyper_param_degree(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42, shuffle=True)

    svc = SVM()

    # svc.train_SVM(Xtrain, Ytrain, f'svm_{DATASET_NAME}')

    #svc.get_accuracy(DATASET_NAME, Xtest, Ytest)
    
    #svc.get_precision_and_recall_scores(DATASET_NAME, Xtest, Ytest)

    # svc.plot_learning_curve(Xtrain,Ytrain,f'{DATASET_NAME}_svm_learning_curve')

    # svc.hyper_param_kernel(DATASET_NAME,Xtrain, Xtest, Ytrain, Ytest)

    # svc.hyper_param_degree(DATASET_NAME,Xtrain, Xtest, Ytrain, Ytest)

    # svc.plot_iterative_curve(DATASET_NAME, Xtrain, Ytrain)

    svc.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)
    