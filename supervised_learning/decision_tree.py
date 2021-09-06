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

def accuracy_dt(dataset_name, data, labels):
    model = load_saved_model(f'decision_tree_{dataset_name}')
    pred = model.predict(data)
    score = accuracy_score(labels.flatten(), pred.flatten())
    print(f'Decision Tree Accuracy: {score}')

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    
    train_decision_tree(Xtrain, Ytrain, f'decision_tree_{DATASET_NAME}')

    accuracy_dt(DATASET_NAME, Xtest, Ytest)

    #plot_learning_curve(tree.DecisionTreeClassifier(),title="Decision Tree Learning Curve",X=data[:50000],y=labels[:50000])
