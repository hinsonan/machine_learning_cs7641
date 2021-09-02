from plot_helpers import plot_accuracy, plot_confusion_matrix, plot_learning_curve
from training import load_saved_model
from data_helper import get_breast_cancer_data, get_cs_go_data
from sklearn.model_selection import train_test_split
from training import tree, svm, AdaBoostClassifier, KNeighborsClassifier
import pandas as pd
import numpy as np

data, labels = get_breast_cancer_data()
# split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,labels, test_size=0.33, random_state=42)
model = load_saved_model('neural_net_breast_cancer',True)
pred = model.predict(Xtest)
pred = np.where(pred<0.5,0,1)
#plot_accuracy(pred[:40],Ytest[:40])
#plot_confusion_matrix(pred,Ytest)

plot_learning_curve(tree.DecisionTreeClassifier(),'TEST',Xtrain,Ytrain)