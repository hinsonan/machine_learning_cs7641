from matplotlib import pyplot as plt
import mlrose_hiive
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score

def get_loan_defualt():
    def normalize_with_min_max_scaler(df:pd.DataFrame):
        values = df.values
        scalar = MinMaxScaler()
        scaled_vals = scalar.fit_transform(values)
        return scaled_vals

    df = pd.read_csv('data/loan_defaulter.csv')
    df = df.drop(['RowNumber','CustomerId','Surname'],axis=1)
    df['Geography'] = LabelEncoder().fit_transform(df['Geography'])
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    vals = normalize_with_min_max_scaler(df)
    data = vals[:,:-1]
    labels = vals[:,-1]
    return data, labels

def plot_fig(train_loss, val_loss, figname, Yaxis):
    _, axes = plt.subplots(1)
    axes.plot(train_loss)
    axes.plot(val_loss)
    axes.set_ylabel(Yaxis)
    axes.set_xlabel('Iterations')
    axes.set_title("NN Learning Curve")
    axes.grid()
    axes.legend(['training', 'validation'], loc='upper left')
    plt.savefig(f"randomized_optimization/{figname}")

def train_net(Xtrain, Xtest, Ytrain, Ytest, algorithm):
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain,Ytrain, test_size=0.2, random_state=42, shuffle=True)
    num_epochs = 100
    train_losses, val_losses, train_accuracies, val_accuracies, train_precisions, val_precisions, train_recalls, val_recalls = [],[],[],[],[],[],[],[]
    for i in range(num_epochs):

        model = mlrose_hiive.NeuralNetwork(hidden_nodes=[8,4], activation='relu',algorithm=algorithm,
                                        max_iters=i,learning_rate=0.001, early_stopping=True, max_attempts=500,
                                        is_classifier=True, pop_size=300,mutation_prob=0.4, restarts=50)
        
        model.fit(Xtrain,Ytrain)

        # calculate losses
        train_pred = model.predict(Xtrain)
        loss = log_loss(Ytrain, train_pred)
        val_pred = model.predict(Xval)
        val_loss = log_loss(Yval,val_pred)

        # calculate accuracy
        train_accuracy = accuracy_score(Ytrain,train_pred)
        val_accuracy = accuracy_score(Yval, val_pred)

        # calculate precision
        train_precision = precision_score(Ytrain,train_pred)
        val_precision = precision_score(Yval, val_pred)

        # calculate recall
        train_recall = recall_score(Ytrain,train_pred)
        val_recall = recall_score(Yval, val_pred)

        train_losses.append(loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)

        print(f'iteration {i}: training loss {loss:.3f} validation loss {val_loss:.3f} training accuracy {train_accuracy:.3f} validation accuracy {val_accuracy:.3f}')

    # get test results and metrics
    test_pred = model.predict(Xtest)
    test_accuracy = accuracy_score(Ytest,test_pred)
    test_precision = precision_score(Ytest,test_pred)
    test_recall = recall_score(Ytest,test_pred)

    plot_fig(train_losses, val_losses, f'NN_{algorithm}_loss', "Log Loss")
    plot_fig(train_accuracies, val_accuracies, f'NN_{algorithm}_accuracy', "Accuracy")
    results = {"training_losses": train_losses, "validation_losses": val_losses,
               "training_accuracies": train_accuracies, "validation_accuracies": val_accuracies,
               "training_precisions": train_precisions, "validation_precisions": val_precisions,
               "training_recalls": train_recalls, "validation_recalls": val_recalls,
               "test_metrics":{"accuracy":test_accuracy,"precision":test_precision,"recall":test_recall}}
    with open(f"randomized_optimization/NN_{algorithm}_metrics.json", 'w') as f:
        json.dump(results, f, indent=4)



data, labels = get_loan_defualt()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42, shuffle=True)

#['random_hill_climb', 'simulated_annealing','genetic_alg', 'gradient_descent']

# train_net(Xtrain, Xtest, Ytrain, Ytest,'random_hill_climb')
# train_net(Xtrain, Xtest, Ytrain, Ytest,'simulated_annealing')
train_net(Xtrain, Xtest, Ytrain, Ytest,'genetic_alg')

