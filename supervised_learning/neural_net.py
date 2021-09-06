from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_helper import get_data, load_saved_model
from plot_helpers import plot_neural_net_history_accuracy, plot_neural_net_history_loss
import numpy as np
import pickle, yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

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

def train_nn(Xtrain,Xtest,Ytrain,Ytest, model_name):
    model = get_model(Xtrain.shape[-1])
    model.summary()
    history = model.fit(Xtrain,Ytrain, batch_size=128, epochs=300, validation_data=(Xtest,Ytest))
    with open(f'supervised_learning/models/{model_name}_history', 'wb') as f:
        pickle.dump(history.history, f)
    model.save(f'supervised_learning/models/{model_name}')

def accuracy_nn(dataset_name, data, labels):
    model = load_model(f'supervised_learning/models/neural_net_{dataset_name}')
    pred = model.predict(data)
    pred = np.where(pred<0.5,0,1)
    score = accuracy_score(labels.flatten(), pred.flatten())
    print(f'Nueral Net Accuracy: {score}')

def learning_curve(dataset_name):
    history = load_saved_model(f'neural_net_{dataset_name}_history')
    plot_neural_net_history_accuracy(history)
    plot_neural_net_history_loss(history)

if __name__ == '__main__':
    with open('supervised_learning/dataset_config.yml','r') as f:
            config = yaml.load(f)

    DATASET_NAME = config[config['Active_Set']]['name']

    data, labels = get_data()
    # split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

    train_nn(Xtrain, Xtest, Ytrain, Ytest, f'neural_net_{DATASET_NAME}')

    accuracy_nn(DATASET_NAME, Xtest, Ytest)

    learning_curve(DATASET_NAME)
