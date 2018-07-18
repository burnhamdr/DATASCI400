#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:55:53 2018

@author: burnhamd
"""
#Import statements
import pandas as pd
import numpy as np
from sklearn import preprocessing
from copy import deepcopy

'''Function checks if a column of data in a dataframe contains binary information.
Takes in column name string, and dataframe object. Returns boolean of whether
the dataframe column is binary.'''
def notBinary(df, column):
        u = df[column].unique().tolist()#grabs unique entries in dataframe column
        has_nan = df[column].isnull().values.any() #checks if column has NaN
        #checks if the column contains binary data, and checks
        #if a column is binary with nan values as well
        notBinary = not (((len(u) == 2) & ((0 in u) & (1 in u))) or 
                         ((len(u) == 3) & ((0 in u) & (1 in u) & has_nan)))
        return notBinary

#Functions to normalize and split the data set
def normalize(df, h): #df = dataframe column, h = column header
    #creates array of the values of the dataframe column
    x = df[[h]].values.astype(float)
    # Create a minimum and maximum processor object
    z_scaler = preprocessing.StandardScaler()
    # Create an object to transform the data to fit minmax processor
    x_scaled = z_scaler.fit_transform(x)
    return x_scaled

def split_dataset(df, train_targ, test_targ, r): #split a dataset in matrix format, using a given ratio for the testing set
	N = len(df)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	nt = N - n # number of elements in training sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = df[ind_,:-1] # training features
	XX = df[ind,:-1] # testing features
	Y = df[ind_,-1] # training targets
	YY = df[ind,-1] # testing targests
	return X, XX, Y, YY
                
##Loading dataset##
#URL of the data set
df1 = pd.read_csv('Data_Cortex_Nuclear.csv')
#create new normalized dataframe
df2 = deepcopy(df1)
for h in headers:
    if notBinary(df2, h):
        n = normalize(df2, h)
        df2[h] = n

df2_m = df2.values
r = 0.2 # ratio of test data over all data
X, XX, Y, YY = split_dataset(df2_m, r)
#Choice of classifier with parameters
#Output of your predicted and actual values
#Comments explaining the code blocks.  