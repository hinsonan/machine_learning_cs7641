from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_helper import get_data, load_saved_model
from plot_helpers import plot_learning_curve
import pickle, yaml

def train_decision_tree(Xtrain,Ytrain, model_name):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrain,Ytrain)
    with open(f'supervised_learning/models/{model_name}', 'wb') as f:
        pickle.dump(clf, f)

def accuracy_experiment():
    model = load_saved_model('decision_tree_cs_go')
    pred = model.predict(Xtest)
    score = accuracy_score(Ytest.flatten(), pred.flatten())
    print(score)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    do_training=False
    do_accuracy=False
    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    if do_training:
        train_decision_tree(Xtrain, Ytrain, f'decision_tree_{DATASET_NAME}')
    if do_accuracy:
        accuracy_experiment()
    plot_learning_curve(tree.DecisionTreeClassifier(),title="Decision Tree Learning Curve",X=data[:50000],y=labels[:50000])
