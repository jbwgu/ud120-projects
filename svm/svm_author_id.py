#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
#sys.path.append("../tools/")
sys.path.insert(0, 'C:/Users/WorkStation/Documents/GitHub/ud120-projects/tools')
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
import warnings 
warnings.filterwarnings('ignore')
from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=10000)

features_train = features_train[:len(features_train)/2]
labels_train = labels_train[:len(labels_train)/2]

t0 = time()
svc.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
pred = svc.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("Accuarcy: ", acc)

def submitAccuracy():
    return acc

# export predictions for 10th, 26th, 50th element
print("10th: ", pred[10])
print("26th: ", pred[26])
print("50th: ", pred[50])

# count number of emails predicted to be from Chris
print("Chris: ", sum(pred))

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
