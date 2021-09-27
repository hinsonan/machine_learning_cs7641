from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from data_helper import get_data, load_saved_model
from plot_helpers import plot_learning_curve, plot_multiple_learning_curves
import pickle, yaml
import numpy as np

class Booster():

    def __init(self):
        pass

    def train_booster(self, Xtrain, Ytrain, model_name):
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(Xtrain,Ytrain)
        with open(f'supervised_learning/models/{model_name}', 'wb') as f:
            pickle.dump(clf, f)

    @staticmethod
    def get_accuracy(dataset_name, data, labels):
        model = load_saved_model(f'booster_{dataset_name}')
        pred = model.predict(data)
        score = accuracy_score(labels.flatten(), pred.flatten())
        print(f'Boosting Accuracy: {score}')
        return score

    @staticmethod
    def get_precision_and_recall_scores(dataset_name, data, labels):
        model = load_saved_model(f'booster_{dataset_name}')
        pred = model.predict(data)
        p_score = precision_score(labels.flatten(), pred.flatten())
        r_score = recall_score(labels.flatten(), pred.flatten())
        print(f'Booster Precision: {p_score}')
        print(f'Booster Recall: {r_score}')
        return p_score, r_score

    def hyper_param_estimators(self, dataset_name, train_x, test_x, train_y, test_y):
        for i in [1,2,3,4,5]:
            clf = AdaBoostClassifier(n_estimators=i, random_state=0)
            clf.fit(train_x,train_y)
            acc_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="accuracy")
            val_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="precision")
            rec_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="recall")
            print(f'boost Accuracy with estimators {i}: {np.mean(acc_scores)}')
            print(f'boost Precision with estimators {i}: {np.mean(val_scores)}')
            print(f'boost Recall with estimators {i}: {np.mean(rec_scores)}')
        
        # plot multiple learning curves
        estimators = [AdaBoostClassifier(n_estimators=1, random_state=0),
                    AdaBoostClassifier(n_estimators=10, random_state=0),
                    AdaBoostClassifier(n_estimators=50, random_state=0),
                    AdaBoostClassifier(n_estimators=100, random_state=0)]
        plot_multiple_learning_curves(estimators, hyper_param_key='n_estimators',title='Booster Performance with variable learners',X=train_x, y=train_y, filename=f'{dataset_name}_booster_numlearners_learning_curve')

    def hyper_param_depth(self, dataset_name, train_x, test_x, train_y, test_y):
        for i in [2,3,4,7]:
            clf = AdaBoostClassifier(n_estimators=5, random_state=0, base_estimator=tree.DecisionTreeClassifier(max_depth=i))
            acc_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="accuracy")
            val_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="precision")
            rec_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="recall")
            print(f'boost Accuracy with depth {i}: {np.mean(acc_scores)}')
            print(f'boost Precision with depth {i}: {np.mean(val_scores)}')
            print(f'boost Recall with depth {i}: {np.mean(rec_scores)}')
        
    
    def plot_learning_curve(self, data, labels, figname):
        plot_learning_curve(AdaBoostClassifier(n_estimators=5, random_state=0, base_estimator=tree.DecisionTreeClassifier(max_depth=3)),title="AdaBoost Learning Curve",X=data,y=labels,filename=figname)

    def gather_metrics(self, train_x, test_x, train_y, test_y):
        clf = AdaBoostClassifier(n_estimators=5, random_state=0, base_estimator=tree.DecisionTreeClassifier(max_depth=3))
        clf.fit(train_x,train_y)
        pred = clf.predict(test_x)
        score = accuracy_score(test_y.flatten(), pred.flatten())
        p_score = precision_score(test_y.flatten(), pred.flatten())
        r_score = recall_score(test_y.flatten(), pred.flatten())
        print(f'boost Accuracy: {score}')
        print(f'boost Precision: {p_score}')
        print(f'boost Recall: {r_score}')

    def run_all(self,dataset,Xtrain, Xtest, Ytrain, Ytest):
        self.plot_learning_curve(Xtrain, Ytrain, f'{dataset}_booster_learning_curve')
        self.hyper_param_estimators(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.hyper_param_depth(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)


if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42, shuffle=True)

    booster = Booster()

    # booster.train_booster(Xtrain, Ytrain, f'booster_{DATASET_NAME}')

    #booster.get_accuracy(DATASET_NAME, Xtest, Ytest)

    #booster.get_precision_and_recall_scores(DATASET_NAME, Xtest, Ytest)

    booster.plot_learning_curve(Xtrain, Ytrain, f'{DATASET_NAME}_booster_learning_curve')

    # booster.hyper_param_estimators(DATASET_NAME, Xtrain, Xtest, Ytrain, Ytest)

    # booster.hyper_param_depth(DATASET_NAME, Xtrain, Xtest, Ytrain, Ytest)

    booster.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)