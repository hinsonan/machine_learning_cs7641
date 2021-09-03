from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_helper import get_breast_cancer_data, get_cs_go_data

import numpy as np

def get_model(input_shape):
    inputs = Input(shape=(input_shape))
    layer = Dense(32, activation='relu')(inputs)
    layer = Dense(16, activation='relu')(layer)
    layer = Dense(4, activation='relu')(layer)
    output = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs, output)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,loss='binary_crossentropy')
    return model

def train_nn(Xtrain,Xtest,Ytrain,Ytest, model_name):
    model = get_model(Xtrain.shape[-1])
    model.summary()
    model.fit(Xtrain,Ytrain, batch_size=128, epochs=300, validation_data=(Xtest,Ytest))
    model.save(f'supervised_learning/models/{model_name}')

data, labels = get_cs_go_data()
# split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)

#train_nn(Xtrain, Xtest, Ytrain, Ytest, 'neural_net_cs_go')

model = load_model('supervised_learning/models/neural_net_cs_go')
pred = model.predict(Xtest)
pred = np.where(pred<0.5,0,1)
score = accuracy_score(Ytest.flatten(), pred.flatten())
print(score)
