#!/usr/bin/env python
#-*- coding: utf-8 -*-

from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def csv_to_vector(filepath):
    vec = genfromtxt(filepath, skip_header=True, delimiter=',')
    return vec[:, 1:], vec[:, 0] 

#########################################################
# Evaluate learning                                     #
#########################################################

def train(x_train, y_train):

    print 'Learning in progress (%s)...' % len(y_train)
    
    # RandomForest: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    clf = RandomForestClassifier(
            # Default parameters
            bootstrap=True, class_weight=None,
            max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            oob_score=False, random_state=0, verbose=0, warm_start=False,
            # criterion
            criterion='gini',
            # criterion='entropy',
            n_jobs=3,
            # number of estimators - ususally bigger the forest the better, there is small chance of overfitting here
            n_estimators=100,
            # max depth of each tree (default none, leading to full tree) - reduction of the maximum depth helps fighting with overfitting
            max_depth=None,
            # max features per split (default sqrt(d)) - you might one to play around a bit as it significantly alters behaviour of the whole tree. 
            # sqrt heuristic is usually a good starting point but an actual sweet spot might be somewhere else
            max_features=None
            )
    # MLP
    # clf = MLPClassifier(hidden_layer_sizes = (500,), activation='relu', solver='lbfgs')
    return clf.fit(x_train, y_train)

def verify(classifier, x_test, y_test):
    print 'Verify classifier'
    y_pred_proba = classifier.predict_proba(x_test)
    y_pred = [y.tolist().index(max(y)) for y in y_pred_proba]
    print 'SCORE: %f' % (sum(y_test == y_pred)/float(len(y_pred)))

#########################################################
# Main                                                  #
#########################################################

def main():
    data, labels = csv_to_vector('data/train.csv')
    # DATA = 80% TRAIN + 10% TEST + 10% VALIDATION
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.30, random_state=42)
    data_validation, data_test, labels_validation, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)
    
    # Learning on our training dataset
    cls = train(data_train, labels_train.astype(int))

    # Verify classifier on our testing dataset
    verify(cls, data_test, labels_test.astype(int))
    
main()

