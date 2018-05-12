'''
Nickallant
tallant.explore

Thisile contains functions to.

1.Read/Load Data
2.Explore Data
3.re-Process and Clean Data
'''
import re
import os
import random
import pandas as pd
import numpy as np
from warnings import warn

from utils import snakify

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder

#importlotly.plotly as py
#importlotly.graph_objs as go

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

def snakify_cols(df, verbose=True):
    '''
    Takes in a dataframe and alters the columns/features
    to snake_case inplace.
    '''
    cols = {col:snakify(col, verbose=verbose) for col in df.columns}
    df.rename(columns=cols, inplace=True)

def get_var_category(df, feature):
    '''Returns either Numerical, Date, Text, or Categorical'''
    unique_count = df[feature].nunique(dropna=False)
    total_count = len(df[feature])
    if pd.api.types.is_numeric_dtype(df[feature]):
        return 'Numerical'
    elif pd.api.types.is_datetime64_dtype(df[feature]):
        return 'Date'
    elif unique_count == total_count:
        return 'Text (Unique)'
    else:
        return 'Categorical'

# Look into Imputer classes from sk-learn
def replace_na_random(feature, lower, upper):
    '''
    Replaces any Null values in a feature with a random int
    between the lower and upper bounds
    '''
    pass

def nan_scan(feature):
    if df[feature].isnull().values.any():
        return 'Has Nans' 
    return 'All Good'

def bound_feature(feature):
    '''why did you make this?''' 
    foi = df[feature]
    ok_vals = df[abs(df[feature]) < 1][feature]
    foi = foi.where(abs(foi) < 1, ok_vals.mean())
    df[feature] = foi

#ev to-do
#dealingith outliers, 
#discretizingontinuous/categorical data


#plotistograms/kde plots of distributions, 
#scatterlot to show relationship between two vars, etc)
