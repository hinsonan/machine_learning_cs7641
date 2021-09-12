from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from data_helper import get_data, load_saved_model
from plot_helpers import plot_learning_curve, plot_multiple_learning_curves
import pickle, yaml

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
            pred = clf.predict(test_x)
            score = accuracy_score(test_y.flatten(), pred.flatten())
            p_score = precision_score(test_y.flatten(), pred.flatten())
            r_score = recall_score(test_y.flatten(), pred.flatten())
            print(f'SVM Accuracy with neighbor {i}: {score}')
            print(f'SVM Precision with neighbor {i}: {p_score}')
            print(f'SVM Recall with neighbor {i}: {r_score}')
        
        # plot multiple learning curves
        estimators = [svm.SVC(kernel='rbf'),
                    svm.SVC(kernel='poly'),
                    svm.SVC(kernel='linear'),
                    svm.SVC(kernel='sigmoid')]
        plot_multiple_learning_curves(estimators, hyper_param_key='kernel',title='SVM performance using different kernels',X=train_x, y=train_y, filename=f'{dataset_name}_svm_multi_learning_curve')

    def plot_learning_curve(self, data, labels):
        plot_learning_curve(svm.SVC(kernel='rbf'),title="KNN Learning Curve",X=data,y=labels)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    svc = SVM()

    #svc.train_SVM(Xtrain, Ytrain, f'svm_{DATASET_NAME}')

    #svc.get_accuracy(DATASET_NAME, Xtest, Ytest)
    
    #svc.get_precision_and_recall_scores(DATASET_NAME, Xtest, Ytest)

    #svc.plot_learning_curve(Xtrain,Ytrain)

    svc.hyper_param_kernel(DATASET_NAME,Xtrain, Xtest, Ytrain, Ytest)
    