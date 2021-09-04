from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_helper import get_data, load_saved_model
import pickle, yaml

def train_SVM(Xtrain,Ytrain, model_name):
    clf = svm.SVC(kernel='rbf',verbose=True)
    clf.fit(Xtrain,Ytrain)
    with open(f'supervised_learning/models/{model_name}', 'wb') as f:
        pickle.dump(clf, f)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    train_SVM(Xtrain, Ytrain, f'svm_{DATASET_NAME}')

    model = load_saved_model(f'svm_{DATASET_NAME}')
    pred = model.predict(Xtest)
    score = accuracy_score(Ytest.flatten(), pred.flatten())
    print(score)