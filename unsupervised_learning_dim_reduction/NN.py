import re
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score,precision_score,recall_score
from data_helpers import get_cs_go_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from plot_helpers import plot_neural_net_history_accuracy,plot_neural_net_history_loss
import numpy as np
import json

class NN:

    def __init__(self) -> None:
        self.dr_algorithms = {
            'pca': PCA(n_components=50),
            'ica': FastICA(n_components=86, random_state=0),
            'rp': GaussianRandomProjection(n_components=70),
            'lda': LinearDiscriminantAnalysis()
        }
        self.clustering_algorithms = {
            'kmeans': KMeans(n_clusters=4, max_iter=500, random_state=0),
            'gmm': GaussianMixture(n_components=4, max_iter=500, random_state=0)
        }
        self.data, self.labels = get_cs_go_data()
        self.eval_data = {}

    def create_model_dr(self,input_dim):
        inputs = Input(shape=(input_dim))
        layer = Dense(50, activation='relu')(inputs)
        layer = Dense(40, activation='relu',activity_regularizer=regularizers.l1_l2())(layer)
        layer = Dense(16, activation='relu',activity_regularizer=regularizers.l1_l2())(layer)
        output = Dense(1, activation="sigmoid")(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def create_model_cluster(self,input_dim):
        inputs = Input(shape=(input_dim))
        layer = Dense(312, activation='relu')(inputs)
        layer = Dense(212, activation='relu')(inputs)
        layer = Dense(128, activation='relu')(layer)
        layer = Dense(64, activation='relu')(layer)
        layer = Dense(32, activation='relu')(layer)
        layer = Dense(16, activation='relu')(layer)
        layer = Dense(8, activation='relu')(layer)
        layer = Dense(4, activation='relu')(layer)
        output = Dense(1, activation="sigmoid")(layer)
        model = Model(inputs, output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def write_out_eval(self,file_name,dir='nn_dr_data'):
        with open(f'unsupervised_learning_dim_reduction/charts/{dir}/{file_name}','w') as f:
            json.dump(self.eval_data,f,indent=4)
        self.eval_data = {}


    def evaluate_model(self, model, train_data, train_labels, val_data, val_labels, test_data, test_labels,algorithm):
        # eval train data
        pred = model.predict(train_data)
        pred = np.where(pred<0.5,0,1)
        acc = accuracy_score(train_labels,pred)
        precision = precision_score(train_labels,pred)
        recall = recall_score(train_labels,pred)
        self.eval_data[algorithm]['training'] = {'Accuracy': acc,'Precision':precision,'Recall':recall}
        # eval validation data
        pred = model.predict(val_data)
        pred = np.where(pred<0.5,0,1)
        acc = accuracy_score(val_labels,pred)
        precision = precision_score(val_labels,pred)
        recall = recall_score(val_labels,pred)
        self.eval_data[algorithm]['validation'] = {'Accuracy': acc,'Precision':precision,'Recall':recall}
        # eval test data
        pred = model.predict(test_data)
        pred = np.where(pred<0.5,0,1)
        acc = accuracy_score(test_labels,pred)
        precision = precision_score(test_labels,pred)
        recall = recall_score(test_labels,pred)
        self.eval_data[algorithm]['testing'] = {'Accuracy': acc,'Precision':precision,'Recall':recall}
        

    def run_dr_net_test(self,dr_algorithm:str):
        if dr_algorithm == 'lda':
            reduced_data = self.dr_algorithms[dr_algorithm].fit_transform(self.data,self.labels)
        else:
            reduced_data = self.dr_algorithms[dr_algorithm].fit_transform(self.data)
        
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(reduced_data,self.labels, test_size=0.33, random_state=42, shuffle=True)

        Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain,Ytrain, test_size=0.2, random_state=42, shuffle=True)

        model = self.create_model_dr(Xtrain[0].shape[0])

        history = model.fit(Xtrain,Ytrain,batch_size=128, epochs=100,validation_data=(Xval,Yval))

        plot_neural_net_history_accuracy(history.history,f'{dr_algorithm}_accuracy',dir='nn_dr_data')
        plot_neural_net_history_loss(history.history,f'{dr_algorithm}_loss',dir='nn_dr_data')

        self.eval_data[dr_algorithm] = {}
        self.evaluate_model(model,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,dr_algorithm)

    def run_clustering_net_test(self,clustering_algorithm:str):
        if clustering_algorithm == 'kmeans':
            kmeans = self.clustering_algorithms[clustering_algorithm].fit(self.data)
            labels = kmeans.labels_
            labels = np.array(labels).reshape(-1,1)
            labels = MinMaxScaler().fit_transform(labels)
            clustered_labeled_data = np.concatenate((self.data,labels),axis=1)
        else:
            gmm = self.clustering_algorithms[clustering_algorithm].fit(self.data)
            labels = gmm.predict(self.data)
            labels = np.array(labels).reshape(-1,1)
            labels = MinMaxScaler().fit_transform(labels)
            clustered_labeled_data = np.concatenate((self.data,labels),axis=1)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(clustered_labeled_data,self.labels, test_size=0.33, random_state=42, shuffle=True)

        Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain,Ytrain, test_size=0.2, random_state=42, shuffle=True)

        model = self.create_model_cluster(Xtrain[0].shape[0])

        history = model.fit(Xtrain,Ytrain,batch_size=128, epochs=100,validation_data=(Xval,Yval))

        plot_neural_net_history_accuracy(history.history,f'{clustering_algorithm}_accuracy',dir='nn_cluster_data')
        plot_neural_net_history_loss(history.history,f'{clustering_algorithm}_loss',dir='nn_cluster_data')

        self.eval_data[clustering_algorithm] = {}
        self.evaluate_model(model,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,clustering_algorithm)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    net_test = NN()
    # Run the DR test
    # net_test.run_dr_net_test('pca')
    # net_test.run_dr_net_test('ica')
    # net_test.run_dr_net_test('rp')
    # net_test.run_dr_net_test('lda')
    # net_test.write_out_eval('nn_dr_metrics',dir='nn_dr_data')

    # Run the clustering nets
    net_test.run_clustering_net_test('kmeans')
    net_test.run_clustering_net_test('gmm')
    net_test.write_out_eval('nn_clustering_metrics',dir='nn_cluster_data')