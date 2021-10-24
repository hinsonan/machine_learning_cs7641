import re
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import accuracy_score,precision_score,recall_score
from data_helpers import get_cs_go_data
from sklearn.model_selection import train_test_split
from plot_helpers import plot_neural_net_history_accuracy,plot_neural_net_history_loss
import numpy as np
import json

class NN:

    def __init__(self) -> None:
        self.dr_algorithms = {
            'pca': PCA(n_components=40),
            'ica': FastICA(n_components=40),
            'rp': GaussianRandomProjection(n_components=50),
            'lda': LinearDiscriminantAnalysis()
        }
        self.data, self.labels = get_cs_go_data()
        self.eval_data = {}

    def create_model(self,input_dim):
        inputs = Input(shape=(input_dim))
        layer = Dense(32, activation='relu')(inputs)
        layer = Dense(32, activation='relu')(layer)
        layer = Dense(32, activation='relu')(layer)
        layer = Dense(4, activation='relu')(layer)
        output = Dense(1, activation="sigmoid")(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def write_out_eval(self,file_name,dir='nn_dr_data'):
        with open(f'unsupervised_learning_dim_reduction/charts/{dir}/{file_name}','w') as f:
            json.dump(self.eval_data,f,indent=4)


    def evaluate_model(self, model, train_data, train_labels, val_data, val_labels, test_data, test_labels,dr_algorithm):
        # eval train data
        pred = model.predict(train_data)
        pred = np.where(pred<0.5,0,1)
        acc = accuracy_score(train_labels,pred)
        precision = precision_score(train_labels,pred)
        recall = recall_score(train_labels,pred)
        self.eval_data[dr_algorithm]['training'] = {'Accuracy': acc,'Precision':precision,'Recall':recall}
        # eval validation data
        pred = model.predict(val_data)
        pred = np.where(pred<0.5,0,1)
        acc = accuracy_score(val_labels,pred)
        precision = precision_score(val_labels,pred)
        recall = recall_score(val_labels,pred)
        self.eval_data[dr_algorithm]['validation'] = {'Accuracy': acc,'Precision':precision,'Recall':recall}
        # eval test data
        pred = model.predict(test_data)
        pred = np.where(pred<0.5,0,1)
        acc = accuracy_score(test_labels,pred)
        precision = precision_score(test_labels,pred)
        recall = recall_score(test_labels,pred)
        self.eval_data[dr_algorithm]['testing'] = {'Accuracy': acc,'Precision':precision,'Recall':recall}
        

    def run_net_test(self,dr_algorithm:str):
        if dr_algorithm == 'lda':
            reduced_data = self.dr_algorithms[dr_algorithm].fit_transform(self.data,self.labels)
        else:
            reduced_data = self.dr_algorithms[dr_algorithm].fit_transform(self.data)
        
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(reduced_data,self.labels, test_size=0.33, random_state=42, shuffle=True)

        Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain,Ytrain, test_size=0.2, random_state=42, shuffle=True)

        model = self.create_model(Xtrain[0].shape[0])

        history = model.fit(Xtrain,Ytrain,batch_size=128, epochs=100,validation_data=(Xval,Yval))

        plot_neural_net_history_accuracy(history.history,f'{dr_algorithm}_accuracy',dir='nn_dr_data')
        plot_neural_net_history_loss(history.history,f'{dr_algorithm}_loss',dir='nn_dr_data')

        self.eval_data[dr_algorithm] = {}
        self.evaluate_model(model,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,dr_algorithm)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    net_test = NN()
    net_test.run_net_test('pca')
    net_test.run_net_test('ica')
    net_test.run_net_test('rp')
    net_test.run_net_test('lda')
    net_test.write_out_eval('nn_dr_metrics',dir='nn_dr_data')