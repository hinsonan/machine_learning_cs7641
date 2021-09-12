from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from data_helper import get_data, load_saved_model
from plot_helpers import plot_learning_curve
import pickle, yaml

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
    
    def plot_learning_curve(self, data, labels):
        plot_learning_curve(AdaBoostClassifier(n_estimators=100, random_state=0),title="AdaBoost Learning Curve",X=data,y=labels)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    booster = Booster()

    #booster.train_booster(Xtrain, Ytrain, f'booster_{DATASET_NAME}')

    booster.get_accuracy(DATASET_NAME, Xtest, Ytest)

    #booster.get_precision_and_recall_scores(DATASET_NAME, Xtest, Ytest)

    #booster.plot_learning_curve(Xtrain, Ytrain)