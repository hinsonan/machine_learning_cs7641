from plot_helpers import plot_learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from data_helper import get_data, load_saved_model
import pickle, yaml

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

    def plot_learning_curve(self, data, labels):
        plot_learning_curve(KNeighborsClassifier(n_neighbors=2),title="KNN Learning Curve",X=data,y=labels)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    knn = KNN()

    #knn.train_knn(Xtrain, Ytrain, f'knn_{DATASET_NAME}')

    knn.get_accuracy(DATASET_NAME, Xtest, Ytest)

    knn.get_precision_and_recall_scores(DATASET_NAME, Xtest, Ytest)

    #knn.plot_learning_curve(Xtrain,Ytrain)