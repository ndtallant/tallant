'''
Nick Tallant
tallant.explorer

This file contains functions to.

1.Read/Load Data
2.Explore Data
3.Pre-Process and Clean Data
'''
import re
import os
import random
import numpy as np
import pandas as pd
from warnings import warn
from scipy.stats import boxcox

#from utils import snakify

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import label_binarize

import seaborn as sns
#import plotly.plotly as py
#import plotly.graph_objs as go
import matplotlib.pyplot as plt


def load_data(data, verbose=True):
    '''Loads in data to a dataframe.'''
    if data.endswith('csv'):
        df =  pd.read_csv(data)
    elif data.endswith('json'):
        df = pd.read_json(data)
    else:
        raise ValueError('I need a csv or json' )
    snakify_cols(df) 
    return df

def factor_df(df):
    for feature in df.columns:
        if df[feature].dtype == 'O' and len(df[feature].unique()) > 2:
            df[feature] = df[feature].factorize()[0]

def snakify_cols(df, verbose=True):
   '''
   Takes in a dataframe and alters the columns/features
   to snake_case inplace.
   '''
   cols = {col:snakify(col, verbose=verbose) for col in df.columns}
   df.rename(columns=cols, inplace=True)

def scale_numeric(df):
    for feature in df.columns:
        if df[feature].dtype == 'int64':
            try:
                df[feature] = minmax_scale(df[feature])
            except ValueError:
                try:
                    df[feature] = boxcox(df[feature])[0]
                except: # I know ...
                    continue

def boxcox_df(df, feature_list):
    for feature in feature_list:
        df[feature] = boxcox(df[feature])[0] 

def minmax_df(df, feature_list):
    for feature in feature_list:
        df[feature] = minmax_scale(df[feature])

def quick_summary(df):
    '''Shows each column, if it has nans, its type, and an example value''' 
    cols = ['Feature'
            , '% Missing'
            , 'Type'
            , 'Uniques'
            , 'Example'
            ]
    return pd.concat([pd.DataFrame([qsum_helper(df, feature)], columns=cols) 
                      for feature in df.columns], 
                      ignore_index=True).set_index('Feature')
        
def qsum_helper(df, feature):
    '''Returns single row to quick_summary''' 
    return [feature, 
            #nan_scan(df, feature), 
            round(len(df[df[feature].isnull()]) / len(df) * 100), 
            get_var_cat(df, feature),
            len(df[feature].unique()),
            df.loc[df.index[0], feature]
            ]

def get_var_cat(df, feature):
    '''Returns either Numerical, Date, Text, or Categorical'''
    unique_count = df[feature].nunique(dropna=False)
    total_count = len(df[feature])
    
    if len(df[feature].unique()) == 2:
        return 'Binary'
    if df[feature].dtype == 'O':
        return 'Categorical'
    if pd.api.types.is_numeric_dtype(df[feature]):
        return 'Numerical'
    if pd.api.types.is_datetime64_dtype(df[feature]):
        return 'Date'
    if unique_count == total_count:
        return 'Text (Unique)'

def nan_scan(df, feature):
    '''Scans for NaNs''' 
    if df[feature].isnull().values.any():
        return 'Yes' 
    return ' No'

def make_numeric(df, feature_list):
    '''Makes a given list of features numeric'''
    for feature in feature_list:
        df[feature] = pd.to_numeric(df[feature])

def dumb_cats(df, sparse=False):
    '''Dummifies all categorical features (dtype object)''' 
    dumb_df = pd.DataFrame(index=df.index)
    for feature in df.columns:
        if df[feature].dtype == 'O' and len(df[feature].unique()) > 2:
            dummied = pd.get_dummies(df[feature], prefix=feature, drop_first=True, sparse=sparse)
            dumb_df = dumb_df.join(dummied)
        else:
            dumb_df = dumb_df.join(df[feature])
    return dumb_df 

def flip_a_bin(df, feature):
    '''Reverses the binary coding on a feature'''
    try: 
        pos, neg = df[feature].unique() 
    except ValueError:
        print('Not binary')
    df[feature] = df[feature].apply(lambda x: pos if x == neg else neg)

def binarize(df, true='t', false='f'):
    '''Binarizes every column in a df, can set the true and false labels''' 
    for feat in df.columns:
        if set(df[feat].unique()) == set([false, true]):
            df[feat] = label_binarize(df[feat], classes=[false, true])
            df[feat] = df[feat].astype('category')

def impute_mean(df, feature):
    '''Imputes Nans as the mean, note the input as a series.'''
    #imp = Imputer(copy=False, axis=1)
    #array = df[feature].values.reshape(1, -1)
    #df[feature].replace(df[feature], imp.fit_transform(array)[0], inplace=True)
    mn = df[feature].dropna().mean()
    df[feature].fillna(mn, inplace=True)

def replace_na_random(feature, lower, upper):
    '''
    Replaces any Null values in a feature with a random int
    between the lower and upper bounds
    '''
    # Look into Imputer classes from sk-learn
    pass

def bound_feature(feature):
    '''why did you make this?''' 
    foi = df[feature]
    ok_vals = df[abs(df[feature]) < 1][feature]
    foi = foi.where(abs(foi) < 1, ok_vals.mean())
    df[feature] = foi

def plot_numerics(df):
    '''Plots every numeric feature in a df (box and hist)''' 
    for feature in df.columns:
        if df[feature].dtype != 'O' and len(df[feature].unique()) > 2:
            plot_numeric(df, feature)
            
def plot_numeric(df, feature):
    '''
    Given a numeric feature, this function plots a 
    box plot and a distribution.
    '''
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = sns.boxplot(y=df[feature])
    fig.set_title('')
    fig.set_ylabel(feature)
    
    plt.subplot(1, 2, 2)
    fig = sns.distplot(df[feature].dropna())
    fig.set_ylabel('')
    fig.set_xlabel(feature)
    plt.show()

def snakify(feature, verbose=False):
    '''
    Changes a feature label to a useful form, by
    making just about any string into snake_case.

    Example:
        Input: 'Hello There'
        Output: 'hello_there'

        Input: 'What aboutThis'
        Output: 'what_about_this'
    
    Thanks to Greg Lamp for a fun name to this idea, and 
    the few functions that inspired this one.
    https://github.com/yhat/DataGotham2013/
    '''
    
    s0 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', feature).title()
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', s0).replace(' ', '')
    s2 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s1)
    snake_feature = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s2).lower()
    
    if len(snake_feature) > 20 and verbose:
        #warn(f'Rename {snake_feature}') 3.6 is the best
        warn('Rename {}'.format(snake_feature))
    # find a way to get rid of source line for this warning. 
    return snake_feature

# Dev Notes for later
# http://pandas.pydata.org/pandas-docs/stable/groupby.html#iterating-through-groups
