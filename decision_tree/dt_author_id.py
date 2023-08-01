#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
#sys.path.append("../tools/")
sys.path.insert(0, 'C:/Users/WorkStation/Documents/GitHub/ud120-projects/tools')
# sys.path.insert(0, 'C:/Users/jtayl/Documents/GitHub/ud120-projects/tools')
from email_preprocess import preprocess
from sklearn import tree



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

tr = tree.DecisionTreeClassifier(min_samples_split=40)

print("Number of features: ", len(features_train[0]))

t0 = time()
tr.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
pred = tr.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("Accuarcy: ", acc)

def submitAccuracy():
    return acc

#########################################################
