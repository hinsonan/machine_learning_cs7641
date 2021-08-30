from plot_helpers import plot_accuracy
from training import load_saved_model
from data_helper import get_breast_cancer_data, get_cs_go_data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data, labels = get_cs_go_data()
# split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)
model = load_saved_model('neural_net_cs_go2',True)
pred = model.predict(Xtest)
pred = np.where(pred<0.5,0,1)
plot_accuracy(pred[:40],Ytest[:40])