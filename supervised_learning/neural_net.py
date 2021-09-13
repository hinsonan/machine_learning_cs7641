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
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

class NeuralNet():

    def __init__(self):
        pass

    @staticmethod
    def get_model(input_shape):
        inputs = Input(shape=(input_shape))
        layer = Dense(32, activation='relu')(inputs)
        layer = Dense(16, activation='relu')(layer)
        layer = Dense(4, activation='relu')(layer)
        output = Dense(1, activation='sigmoid')(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_nn(self, Xtrain,Xtest,Ytrain,Ytest, model_name):
        model = self.get_model(Xtrain.shape[-1])
        model.summary()
        history = model.fit(Xtrain,Ytrain, batch_size=128, epochs=300, validation_data=(Xtest,Ytest))
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
        inputs = Input(shape=(train_x.shape[-1]))
        layer = Dense(32, activation='tanh')(inputs)
        layer = Dense(16, activation='tanh')(layer)
        layer = Dense(4, activation='tanh')(layer)
        output = Dense(1, activation='sigmoid')(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

        inputs = Input(shape=(train_x.shape[-1]))
        layer = Dense(32, activation='relu')(inputs)
        layer = Dense(16, activation='relu')(layer)
        layer = Dense(4, activation='relu')(layer)
        output = Dense(1, activation='sigmoid')(layer)
        model2 = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model2.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

        history1 = model.fit(train_x,train_y, batch_size=128, epochs=100, validation_data=(test_x,test_y))
        history2 = model2.fit(train_x,train_y, batch_size=128, epochs=100, validation_data=(test_x,test_y))

        plot_multiple_histories(history1.history,history2.history, f'{dataset_name}_nn_activation_learner', ['tanh','relu'])

    def hyper_param_layers(self, dataset_name, train_x, test_x, train_y, test_y):
        inputs = Input(shape=(train_x.shape[-1]))
        layer = Dense(32, activation='relu')(inputs)
        output = Dense(1, activation='sigmoid')(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

        inputs = Input(shape=(train_x.shape[-1]))
        layer = Dense(32, activation='relu')(inputs)
        layer = Dense(16, activation='relu')(layer)
        layer = Dense(4, activation='relu')(layer)
        output = Dense(1, activation='sigmoid')(layer)
        model2 = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model2.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

        history1 = model.fit(train_x,train_y, batch_size=128, epochs=100, validation_data=(test_x,test_y))
        history2 = model2.fit(train_x,train_y, batch_size=128, epochs=100, validation_data=(test_x,test_y))

        plot_multiple_histories(history1.history,history2.history, f'{dataset_name}_nn_layer_learner', ['1 layer','4 layer'])

    def plot_learning_curve(self, dataset_name):
        history = load_saved_model(f'neural_net_{dataset_name}_history')
        plot_neural_net_history_accuracy(history,f'{dataset_name}_nn_accuracy')
        plot_neural_net_history_loss(history, f'{dataset_name}_nn_loss')

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    neural_net = NeuralNet()

    #neural_net.train_nn(Xtrain, Xtest, Ytrain, Ytest, f'neural_net_{DATASET_NAME}')

    #neural_net.get_accuracy(DATASET_NAME, Xtest, Ytest)

    #neural_net.get_precision_and_recall_scores(DATASET_NAME, Xtest, Ytest)

    #neural_net.plot_learning_curve(DATASET_NAME)

    #neural_net.hyper_param_activation(DATASET_NAME, Xtrain, Xtest, Ytrain, Ytest)

    neural_net.hyper_param_layers(DATASET_NAME, Xtrain, Xtest, Ytrain, Ytest)
