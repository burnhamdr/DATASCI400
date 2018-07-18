#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 23:44:45 2018

@author: burnhamd
"""

import pandas as pd
import numpy as np
from Functions import cleanFunctions as cf
from sklearn import preprocessing
from copy import deepcopy

def normalize(df, h): #df = dataframe column, h = column header
    #creates array of the values of the dataframe column
    x = df[[h]].values.astype(float)
    # Create a minimum and maximum processor object
    z_scaler = preprocessing.StandardScaler()
    # Create an object to transform the data to fit minmax processor
    x_scaled = z_scaler.fit_transform(x)
    return x_scaled

df1 = pd.read_csv('Data_Cortex_Nuclear.csv')
#Check to see amount of missing data, and replace using median imputation
#Check variable distributions of numeric data types
#cf.plotDistributions(df1)

#Normalize variables with z-score normalization
df2 = deepcopy(df1)#make copy of raw data dataframe to process
df2 = df2.drop("MouseID", axis=1)
#get stats on outlier and missing value incidence within the dataset
outlier_cols, rows_missing, columns_missing, count = cf.clean(df2)
percent_missing = count/(df1.shape[0] * df1.shape[1])
#fill in missing values with median values
df3 = cf.medianImputation(df2)
#z-score normalization for the dataframe
for h in list(df3):
    if cf.notBinary(df3, h) & cf.is_number(df3, h):
        n = normalize(df3, h)
        df3[h] = n