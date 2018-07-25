#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:05:19 2018

@author: burnhamd
"""
'''This set of functions provides dataframe cleaning functionality.'''
##Import statements##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


'''Function to check if an input dataframe contains at least one string 
that is convertable to a float value. Returns boolean value indicating whether
the input string is convertable.  Used in the case that the string values
in the dataframe are not unicode.'''
def is_number(df, column):
    is_float = np.empty(0)
    for index in df.index.tolist():
        try:
            float(df.loc[index][column])
            is_float = np.append(is_float, True)
        except ValueError:
            is_float = np.append(is_float, False)
    if sum(is_float) >= 1:
        return True
    else:
        return False

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

'''Function that counts the number of null entries in a dataframe, the
number of rows with atleast one null entry, and the number of columns with
atleast one null entry.

Inputs:
    df: dataframe pandas object

Returns:
    count: Total number of missing entries
    null_entries: A numpy array with column one indicating the dataframe row
                    index that has at least one missing value, and column two
                    indicating the number of missing values for that index.
    null_categories: A numpy array with column one indicating the dataframe
                        header that has at least one missing value in its
                        column, and column two indicating the number of missing
                        values for each header.'''
def countMissing(df):
    count = 0
    null_entries = np.empty(0)
    null_categories = np.empty(0)
    for column in df:
        for index in df.index.tolist():
            if((df.loc[index][column] == "?") or (df.loc[index][column] == " ")
            or (df.loc[index][column] == "") or (df.loc[index][column] == "NA")
            or (pd.isna(df.loc[index][column]))):            
                count = count + 1
                null_entries = np.append(null_entries, index)
                null_categories = np.append(null_categories, column)

    #organizes unique indice values that have NaNs, with corresponding counts
    #of NaNs per row.
    ne, ec = np.unique(null_entries, return_counts=True)
    null_entries = np.asarray((ne, ec)).T
    nc, cc = np.unique(null_categories, return_counts=True)
    null_categories = np.asarray((nc, cc)).T
    
    return count, null_entries, null_categories

'''Function that removes outliers by setting them equal to the average of the
non-outliers.  Takes in a dataframe of any length, and removes outliers
of any numeric column that does not contain binary information.  In addition,
if a column contains numeric strings, these are converted to numeric values
and outliers are removed from this data.

Inputs:
    df: Dataframe pandas object
    
Returns:
    outlier_cols: A numpy array with two columns and N rows depending on how
                    many columns contain outlier values.  The first column
                    indicates which headers from the dataframe correspond to
                    columns in the dataframe with outlier values.  The second
                    column indicates the frequency of outlier values per
                    dataframe header.'''
def outlierRemover(df):
    outlier_cols = np.empty(0)#tracks which columns have outliers
    outlier_counts = np.empty(0)
    for column in df:
        if notBinary(df, column): #binary columns excluded from outlier removal operations
            #checks if the column datatype is numeric
            if(df[column].dtype == np.float64 or df[column].dtype == np.int64):
                LimitHi = df[column].mean() + 2.5*df[column].std()#plus 2 std
                LimitLo = df[column].mean() - 2.5*df[column].std()#minus 2 std
                #Create dataframe for values within limits
                good = df[(df[column] >= LimitLo) & (df[column] <= LimitHi)]
                #Create dataframe for values outside limits
                bad = df[(df[column] < LimitLo) | (df[column] > LimitHi)]
                bad_id = bad.index.tolist()#list of outlier indices
                #adds column header to tracker if outliers are present
                if len(bad_id) >= 1:
                    outlier_cols = np.append(outlier_cols, column)
                    outlier_counts = np.append(outlier_counts, len(bad_id))
                # Replace outleiers with the mean of non-outliers
                df.at[bad_id, column] = good[column].mean()
            else:
                n = df[column].str.isnumeric()#flags numeric strings
                #if there is atleast 1 numeric string value, assume that the 
                #dtype of the column should be numeric and then find outliers
                if (np.sum(n) > 0 or is_number(df, column)):
                    #change dtype of the dataframe column to numeric
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                    LimitHi = df[column].mean() + 2*df[column].std()#plus 2 std
                    LimitLo = df[column].mean() - 2*df[column].std()#minus 2 std
                    #Create dataframe for values within limits
                    good = df[(df[column] >= LimitLo) & (df[column] <= LimitHi)]
                    #Create dataframe for values outside limits
                    bad = df[(df[column] < LimitLo) | (df[column] > LimitHi)]
                    bad_id = bad.index.tolist()#list of outlier indices
                    #adds column header to tracker if outliers are present
                    if len(bad_id) >= 1:
                        outlier_cols = np.append(outlier_cols, column)
                        outlier_counts = np.append(outlier_counts, len(bad_id))
                    # Replace outleiers with the mean of non-outliers
                    df.at[bad_id, column] = good[column].mean()
    #packages the outlier column headers together with the frequency of
    #outlier values per column.
    outlier_cols = np.asarray((outlier_cols, outlier_counts)).T
                    
    return outlier_cols

##Funtion that fills in missing values##
'''Function that fills in missing values.  Takes in a dataframe of any length
and removes missing values marked by ?, blank space, or empty entries.  
Returns the number of missing entries, a numpy array of the indices of rows 
with missing entries and the frequency of missing entries per row, and a numpy 
array of the columns with missing entries and the frequency of missing entries per
column.'''
def replaceMissing(df):
    c, rm, cm = countMissing(df)
    for column in df:
        for index in df.index.tolist():
            #finds locations of missing values marked by ? or blank space
            if((df.loc[index][column] == "?") or (df.loc[index][column] == " ")
            or (df.loc[index][column] == "") or (df.loc[index][column] == "NA")):
                df.at[index, column] = np.nan#replaces missing value with NaN
    
    return c, rm, cm

'''Function that cleans the data by first replacing missing values, and 
then removing outliers from the data.  Takes in a dataframe of any length as
an input, and modifies the dataframe in place.  Returns (from outlierRemover)
a numpy array of the column names with outlier entries, a numpy array of 
the indices of rows with missing entries, and a numpy array of the column
names with missing entries.

Inputs:
    df: Dataframe pandas object

Returns:
    outlier_cols: A numpy array with two columns and N rows depending on how
                many columns contain outlier values.  The first column
                indicates which headers from the dataframe correspond to
                columns in the dataframe with outlier values.  The second
                column indicates the frequency of outlier values per
                dataframe header.
    count: Total number of missing entries
    null_entries: A numpy array with column one indicating the dataframe row
                    index that has at least one missing value, and column two
                    indicating the number of missing values for that index.
    null_categories: A numpy array with column one indicating the dataframe
                        header that has at least one missing value in its
                        column, and column two indicating the number of missing
                        values for each header.'''
def clean(df):
    count, rows_missing, columns_missing = replaceMissing(df)
    outlier_cols = outlierRemover(df)
    
    return outlier_cols, rows_missing, columns_missing, count
            
'''Function that plots distributions of each non-binary, numeric dataframe
attribute column.  NaN terms are dropped from the distribution for plotting.'''
def plotDistributions(df):
    for column in df:
        if notBinary(df, column): #binary columns excluded from outlier removal operations
            #checks if the column datatype is numeric
            if(df[column].dtype == np.float64 or df[column].dtype == np.int64):
                dist = df.loc[:, column]
                plt.figure()
                plt.title(column)
                plt.hist(dist.dropna())#ignores NaN values
 
'''Function that removes missing values and replaces outliers with mean values, 
Returns a new dataframe with NaN values removed using median imputation.'''               
def medianImputation(df):
    df_copy = df.copy()
    clean(df_copy)
    for column in df_copy:
        #checks if column is binary to avoid replacing NaN with median and 
        #guessing a binary state when none is indicated.  Also checks that 
        #the column dtype is numeric and thus has a median value.
        if ((notBinary(df_copy, column)) & 
        ((df_copy[column].dtype == np.float64) or 
         (df_copy[column].dtype == np.int64))):
            
            HasNan = np.isnan(df_copy.loc[:,column])
            df_copy.loc[HasNan, column] = np.nanmedian(df_copy.loc[:, column])
            
    return df_copy

'''Function that normalizes each numeric, non-binary column of the inputted
dataframe.  The function takes in a boolean value to indicate whether to 
use a min max normalization or the standard z-score normalization method. 
Returns a new dataframe with normalized values.'''
def normalizer(df, minMax):
    df_copy = df.copy()
    for h in list(df):
        if notBinary(df, h) & is_number(df, h):
            #creates array of the values of the dataframe column
            x = df[[h]].values.astype(float)
            # Create a minimum and maximum processor object
            if minMax:
                minmax_scale = preprocessing.MinMaxScaler()
                x_scaled = minmax_scale.fit_transform(x)
            else:
                z_scaler = preprocessing.StandardScaler()
                # Create an object to transform the data to fit minmax processor
                x_scaled = z_scaler.fit_transform(x)
            df_copy[h] = x_scaled
    return df_copy

# split a dataset in dataframe format, using a given ratio for the testing set
def splitter(df, ratio, seed, pred_col):
    #index object with indices of overall dataframe
    all_ind = pd.Index(df.index.values)
    
    if ratio >= 1: 
        print ("Parameter r needs to be smaller than 1!")
        return
    elif ratio <= 0:
        print ("Parameter r needs to be larger than 0!")
        return
    #random split of the overall dataframe into a test subset
    df_test = df.sample(frac=ratio, random_state=seed)
    #list of the indices of the test dataframe split
    ind_test = df_test.index.values.tolist()
    #boolean array of where test indices are in the overall dataframe
    split_indices = all_ind.isin(ind_test)
    df_train = df[~split_indices] #dataframe containing data to train on
    df_train_out = df_train[pred_col] #output to predict in training dataset
    df_test_out = df_test[pred_col] #output to predict in testing dataset
    #remove output column from training dataset
    df_train = df_train.drop(pred_col, axis = 1)
    #remove output column from testing dataset
    df_test = df_test.drop(pred_col, axis = 1)
    
    #return the testing and training dataset with respective outputs to predict
    return df_test, df_test_out, df_train, df_train_out