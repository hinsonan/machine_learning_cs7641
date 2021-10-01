from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data_helper import get_data, load_saved_model
from plot_helpers import plot_neural_net_history_accuracy, plot_neural_net_history_loss, plot_multiple_histories
import numpy as np
import pickle, yaml
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

class NeuralNet():

    def __init__(self):
        pass

    @staticmethod
    def get_model(input_shape):
        inputs = Input(shape=(input_shape))
        layer = Dense(312, activation='relu')(inputs)
        layer = Dense(212, activation='relu')(inputs)
        layer = Dense(128, activation='relu')(layer)
        layer = Dense(64, activation='relu')(layer)
        layer = Dense(32, activation='relu')(layer)
        layer = Dense(16, activation='relu')(layer)
        layer = Dense(8, activation='relu')(layer)
        layer = Dense(4, activation='relu')(layer)
        # layer = Dense(8, activation="swish")(inputs)
        # layer = Dense(4, activation="swish")(layer)
        output = Dense(1, activation="sigmoid")(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_nn(self, Xtrain,Xtest,Ytrain,Ytest, model_name):
        model = self.get_model(Xtrain.shape[-1])
        model.summary()
        history = model.fit(Xtrain,Ytrain, batch_size=128, epochs=100, validation_data=(Xtest,Ytest))
        with open(f'supervised_learning/models/{model_name}_history', 'wb') as f:
            pickle.dump(history.history, f)
        model.save(f'supervised_learning/models/{model_name}')

    @staticmethod
    def get_accuracy(dataset_name, data, labels):
        model = load_model(f'supervised_learning/models/neural_net_{dataset_name}')
        pred = model.predict(data)
        pred = np.where(pred<0.5,0,1)
        score = accuracy_score(labels.flatten(), pred.flatten())
        print(f'Nueral Net Accuracy: {score}')
        return score

    @staticmethod
    def get_precision_and_recall_scores(dataset_name, data, labels):
        model = load_model(f'supervised_learning/models/neural_net_{dataset_name}')
        pred = model.predict(data)
        pred = np.where(pred<0.5,0,1)
        p_score = precision_score(labels.flatten(), pred.flatten())
        r_score = recall_score(labels.flatten(), pred.flatten())
        print(f'Neural Net Precision: {p_score}')
        print(f'Neural Net Recall: {r_score}')
        return p_score, r_score

    def hyper_param_activation(self, dataset_name, train_x, test_x, train_y, test_y):
        Xtrain, Xval, Ytrain, Yval = train_test_split(train_x,train_y, test_size=0.25, random_state=42, shuffle=True)
        values = {"swish":[], "relu":[], "sigmoid":[], "tanh":[]}
        for i in ["swish", "relu", "sigmoid", "tanh"]:
            inputs = Input(shape=(train_x.shape[-1]))
            layer = Dense(8, activation=i)(inputs)
            layer = Dense(4, activation=i)(layer)
            output = Dense(1, activation=i)(layer)
            model = Model(inputs, output)
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(Xtrain,Ytrain, batch_size=128, epochs=100, validation_data=(Xval,Yval))

            pred = model.predict(Xval)
            pred = np.where(pred<0.5,0,1)
            score = accuracy_score(Yval.flatten(), pred.flatten())
            p_score = precision_score(Yval.flatten(), pred.flatten())
            r_score = recall_score(Yval.flatten(), pred.flatten())
            values[i].append(score)
            values[i].append(p_score)
            values[i].append(r_score)
        for key in values.keys():
            print(f'Nueral Net {key} Accuracy: {values[key][0]}')
            print(f'Neural Net {key} Precision: {values[key][1]}')
            print(f'Neural Net {key} Recall: {values[key][2]}')


    def hyper_param_layers(self, dataset_name, train_x, test_x, train_y, test_y):
        Xtrain, Xval, Ytrain, Yval = train_test_split(train_x,train_y, test_size=0.25, random_state=42, shuffle=True)
        inputs = Input(shape=(train_x.shape[-1]))
        layer = Dense(64, activation='relu')(inputs)
        layer = Dense(8, activation='relu')(layer)
        layer = Dense(4, activation='relu')(layer)
        output = Dense(1, activation='sigmoid')(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

        inputs = Input(shape=(train_x.shape[-1]))
        layer = Dense(8, activation='relu')(inputs)
        layer = Dense(4, activation='relu')(layer)
        output = Dense(1, activation='sigmoid')(layer)
        model2 = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model2.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

        history1 = model.fit(Xtrain,Ytrain, batch_size=128, epochs=100, validation_data=(Xval,Yval))
        history2 = model2.fit(Xtrain,Ytrain, batch_size=128, epochs=100, validation_data=(Xval,Yval))

        pred = model.predict(Xval)
        pred = np.where(pred<0.5,0,1)
        score = accuracy_score(Yval.flatten(), pred.flatten())
        p_score = precision_score(Yval.flatten(), pred.flatten())
        r_score = recall_score(Yval.flatten(), pred.flatten())
        print(f'Nueral Net 1 Accuracy: {score}')
        print(f'Neural Net 1 Precision: {p_score}')
        print(f'Neural Net 1 Recall: {r_score}')

        pred = model2.predict(Xval)
        pred = np.where(pred<0.5,0,1)
        score = accuracy_score(Yval.flatten(), pred.flatten())
        p_score = precision_score(Yval.flatten(), pred.flatten())
        r_score = recall_score(Yval.flatten(), pred.flatten())
        print(f'Nueral Net 2 Accuracy: {score}')
        print(f'Neural Net 2 Precision: {p_score}')
        print(f'Neural Net 2 Recall: {r_score}')

        plot_multiple_histories(history1.history,history2.history, f'{dataset_name}_nn_layer_learner', ['model 1','model 2'])

    def plot_learning_curve(self, dataset_name, train_x, train_y):
        Xtrain, Xval, Ytrain, Yval = train_test_split(train_x,train_y, test_size=0.25, random_state=42, shuffle=True)
        model = self.get_model(Xtrain.shape[-1])
        history = model.fit(Xtrain,Ytrain, batch_size=128, epochs=100, validation_data=(Xval,Yval))
        plot_neural_net_history_accuracy(history.history,f'{dataset_name}_nn_accuracy')
        plot_neural_net_history_loss(history.history, f'{dataset_name}_nn_loss')

    def gather_metrics(self, Xtrain, Xtest, Ytrain, Ytest):
        inputs = Input(shape=(Xtrain.shape[-1]))
        layer = Dense(8, activation="swish")(inputs)
        layer = Dense(4, activation="swish")(layer)
        output = Dense(1, activation="swish")(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(Xtrain, Ytrain, batch_size=128, epochs=100)

        pred = model.predict(Xtest)
        pred = np.where(pred<0.5,0,1)
        score = accuracy_score(Ytest.flatten(), pred.flatten())
        p_score = precision_score(Ytest.flatten(), pred.flatten())
        r_score = recall_score(Ytest.flatten(), pred.flatten())
        print(f'Nueral Net Accuracy: {score}')
        print(f'Neural Net Precision: {p_score}')
        print(f'Neural Net Recall: {r_score}')

    def run_all(self,dataset, Xtrain, Xtest, Ytrain, Ytest):
        self.plot_learning_curve(Xtrain, Ytrain, f'{dataset}_booster_learning_curve')
        self.hyper_param_layers(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.hyper_param_activation(dataset, Xtrain, Xtest, Ytrain, Ytest)
        self.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42, shuffle=True)

    neural_net = NeuralNet()

    # neural_net.train_nn(Xtrain, Xtest, Ytrain, Ytest, f'neural_net_{DATASET_NAME}')

    #neural_net.get_accuracy(DATASET_NAME, Xtest, Ytest)

    #neural_net.get_precision_and_recall_scores(DATASET_NAME, Xtest, Ytest)

    # neural_net.plot_learning_curve(DATASET_NAME, Xtrain, Ytrain)

    # neural_net.hyper_param_activation(DATASET_NAME, Xtrain, Xtest, Ytrain, Ytest)

    # neural_net.hyper_param_layers(DATASET_NAME, Xtrain, Xtest, Ytrain, Ytest)

    neural_net.gather_metrics(Xtrain, Xtest, Ytrain, Ytest)