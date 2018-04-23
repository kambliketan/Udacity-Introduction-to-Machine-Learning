#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
print "features_train shape:", features_train.shape
print "features_test shape:", features_test.shape

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"
t0 = time()
labels_pred = clf.predict(features_test)
print "testing time:", round(time() - t0, 3), "s"
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, labels_pred)
print acc

# output with min_samples_split = 40
# (15820L, 3785L)
# (1758L, 3785L)
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# training time: 43.979 s
# testing time: 0.04 s
# 0.978953356086

#########################################################


