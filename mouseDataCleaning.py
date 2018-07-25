#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 23:44:45 2018

@author: burnhamd
"""

import pandas as pd
from Functions import cleanFunctions as cf
from copy import deepcopy
import random
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

'''DATA PREPARATION'''
rs = random.seed(15)

df1 = pd.read_csv('Data_Cortex_Nuclear.csv')
#Check variable distributions of numeric data types
#cf.plotDistributions(df1)

df2 = deepcopy(df1)#make copy of raw data dataframe to process
#drops MouseID column as the dataframe index conveys this information
df2 = df2.drop("MouseID", axis=1)

#get stats on outlier and missing value incidence within the dataset
outlier_cols, rows_missing, columns_missing, count = cf.clean(df2)

#indicates percent of dataset that is missing
percent_missing = count/(df1.shape[0] * df1.shape[1])

#fill in missing values with median values
df3 = cf.medianImputation(df2)

#z-score normalization for the dataframe
df4 = cf.normalizer(df3, False)

#drops Genotype, Treatment, and Behavior column as this information is
#conveyed in the Class column
df4 = df4.drop(["Genotype", "Treatment", "Behavior"], axis=1)

#split into train and test dataframes for predictive modeling
df_test, df_test_y, df_train, df_train_y = cf.splitter(df4, 0.4, rs, "class")

'''
""" CLASSIFICATION MODELS """
# Logistic regression classifier
print ('\n\n\nLogistic regression classifier\n')
C_parameter = 50. / len(df_train) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
#####################

#Training the Model
clf = LogisticRegression(C=C_parameter, multi_class=class_parameter, 
                         penalty=penalty_parameter, solver=solver_parameter, 
                         tol=tolerance_parameter)
clf.fit(df_train, df_train_y) 
print ('coefficients:')
# each row of this matrix corresponds to each one of the classes of the dataset
print (clf.coef_)
print ('intercept:')
# each element of this vector corresponds to each one of the classes of the dataset
print (clf.intercept_)

# Apply the Model
print ('predictions for test set:')
print (clf.predict(df_test))
print ('actual class values:')
print (df_)
#####################

# Naive Bayes classifier
print ('\n\nNaive Bayes classifier\n')
nbc = GaussianNB() # default parameters are fine
nbc.fit(X, Y)
print ("predictions for test set:")
print (nbc.predict(XX))
print ('actual class values:')
print (YY)
####################

# k Nearest Neighbors classifier
print ('\n\nK nearest neighbors classifier\n')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
knn.fit(X, Y)
print ("predictions for test set:")
print (knn.predict(XX))
print ('actual class values:')
print (YY)
###################

# Support vector machine classifier
t = 0.001 # tolerance parameter
kp = 'rbf' # kernel parameter
print ('\n\nSupport Vector Machine classifier\n')
clf = SVC(kernel=kp, tol=t)
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)
####################

# Decision Tree classifier
print ('\n\nDecision Tree classifier\n')
clf = DecisionTreeClassifier() # default parameters are fine
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)
####################

# Random Forest classifier
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)
####################
'''