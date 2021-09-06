from plot_helpers import plot_learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_helper import get_data, load_saved_model
import pickle, yaml

def train_knn(Xtrain, Ytrain, model_name):
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(Xtrain,Ytrain)
    with open(f'supervised_learning/models/{model_name}', 'wb') as f:
        pickle.dump(clf, f)

def accuracy_knn(dataset_name, data, labels):
    model = load_saved_model(f'knn_{dataset_name}')
    pred = model.predict(data)
    score = accuracy_score(labels.flatten(), pred.flatten())
    print(f'KNN Accuracy: {score}')

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    train_knn(Xtrain, Ytrain, f'knn_{DATASET_NAME}')

    accuracy_knn(DATASET_NAME, Xtest, Ytest)

    #plot_learning_curve(KNeighborsClassifier(n_neighbors=2),'KNN',Xtrain[:30000],Ytrain[:30000])