from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from data_helper import get_data, load_saved_model
from plot_helpers import plot_learning_curve, plot_multiple_learning_curves
import pickle, yaml
import numpy as np

class DecisionTree():

    def __init__(self):
        pass

    def train_decision_tree(self, Xtrain,Ytrain, model_name):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(Xtrain,Ytrain)
        with open(f'supervised_learning/models/{model_name}', 'wb') as f:
            pickle.dump(clf, f)
    
    @staticmethod
    def get_accuracy(dataset_name, data, labels):
        model = load_saved_model(f'decision_tree_{dataset_name}')
        pred = model.predict(data)
        score = accuracy_score(labels.flatten(), pred.flatten())
        print(f'Decision Tree Accuracy: {score}')
        return score

    @staticmethod
    def get_precision_and_recall_scores(dataset_name, data, labels):
        model = load_saved_model(f'decision_tree_{dataset_name}')
        pred = model.predict(data)
        p_score = precision_score(labels.flatten(), pred.flatten())
        r_score = recall_score(labels.flatten(), pred.flatten())
        print(f'Decision Tree Precision: {p_score}')
        print(f'Decision Tree Recall: {r_score}')
        return p_score, r_score

    def hyper_param_maxdepth(self, dataset_name, train_x, test_x, train_y, test_y):
        for i in [4,5,6,7,8,9,10,15]:
            clf = tree.DecisionTreeClassifier(max_depth=i)
            clf.fit(train_x,train_y)
            acc_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="accuracy")
            val_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="precision")
            rec_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="recall")
            print(f'DT Accuracy with max depth {i}: {np.mean(acc_scores)}')
            print(f'DT Precision with max depth {i}: {np.mean(val_scores)}')
            print(f'DT Recall with max depth {i}: {np.mean(rec_scores)}')
        
        # plot multiple learning curves
        estimators = [tree.DecisionTreeClassifier(max_depth=1),
                    tree.DecisionTreeClassifier(max_depth=2),
                    tree.DecisionTreeClassifier(max_depth=3),
                    tree.DecisionTreeClassifier(max_depth=4),
                    tree.DecisionTreeClassifier(max_depth=5)]
        plot_multiple_learning_curves(estimators, hyper_param_key='max_depth',title='Decision Tree Performance Using Pruning',X=train_x, y=train_y, filename=f'{dataset_name}_dt_max_depth_learning_curve')

    def hyper_param_leafnodes(self, dataset_name, train_x, test_x, train_y, test_y):
        for i in [5,10,15,20,25]:
            clf = tree.DecisionTreeClassifier(max_leaf_nodes=i, max_depth=5)
            acc_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="accuracy")
            val_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="precision")
            rec_scores = cross_val_score(clf,train_x,train_y, cv=5, scoring="recall")
            print(f'DT Accuracy with leadnodes {i}: {np.mean(acc_scores)}')
            print(f'DT Precision with leadnodes {i}: {np.mean(val_scores)}')
            print(f'DT Recall with leadnodes {i}: {np.mean(rec_scores)}')
        
        # plot multiple learning curves
        estimators = [tree.DecisionTreeClassifier(max_leaf_nodes=2),
                    tree.DecisionTreeClassifier(max_leaf_nodes=5),
                    tree.DecisionTreeClassifier(max_leaf_nodes=10),
                    tree.DecisionTreeClassifier(max_leaf_nodes=20),
                    tree.DecisionTreeClassifier(max_leaf_nodes=30)]
        plot_multiple_learning_curves(estimators, hyper_param_key='max_leaf_nodes',title='Decision Tree Performance with limiting leaf nodes',X=train_x, y=train_y, filename=f'{dataset_name}_dt_leaf_nodes_learning_curve')


    def plot_learning_curve(self, data, labels, figname):
        plot_learning_curve(tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=25),title="Decision Tree Learning Curve",X=data,y=labels, filename=figname)

    def gather_metrics(self, train_x, test_x, train_y, test_y):
        clf = tree.DecisionTreeClassifier(max_depth=30, max_leaf_nodes=1100)
        clf.fit(train_x,train_y)
        pred = clf.predict(test_x)
        score = accuracy_score(test_y.flatten(), pred.flatten())
        p_score = precision_score(test_y.flatten(), pred.flatten())
        r_score = recall_score(test_y.flatten(), pred.flatten())
        print(f'DT Accuracy: {score}')
        print(f'DT Precision: {p_score}')
        print(f'DT Recall: {r_score}')

    def run_all(self,dataset,Xtrain, Xtest, Ytrain, Ytest):
        self.plot_learning_curve(Xtrain, Ytrain, f'{dataset}_booster_learning_curve')
        self.hyper_param_maxdepth(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.hyper_param_leafnodes(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42, shuffle=True)

    dt = DecisionTree()
    
    # dt.train_decision_tree(Xtrain, Ytrain, f'decision_tree_{DATASET_NAME}')

    #dt.get_accuracy(DATASET_NAME, Xtest, Ytest)

    #dt.get_precision_and_recall_scores(DATASET_NAME, Xtest, Ytest)

    # dt.plot_learning_curve(Xtrain, Ytrain, f'{DATASET_NAME}_dt_learning_curve')

    # dt.hyper_param_maxdepth(DATASET_NAME, Xtrain, Xtest, Ytrain, Ytest)

    # dt.hyper_param_leafnodes(DATASET_NAME, Xtrain, Xtest, Ytrain, Ytest)

    dt.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)
