'''
Nick Tallant
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

#from utils import snakify

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

def nan_scan(df):
    for feature in df.columns:
        if df[feature].isnull().values.any():
            print(feature, '---> Has Nans') 
        else:
            print(feature, '---> All Good')

def bound_feature(feature):
    '''why did you make this?''' 
    foi = df[feature]
    ok_vals = df[abs(df[feature]) < 1][feature]
    foi = foi.where(abs(foi) < 1, ok_vals.mean())
    df[feature] = foi

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

#dev to-do
#dealingith outliers, 
#discretizingontinuous/categorical data

#plotistograms/kde plots of distributions, 
#scatterlot to show relationship between two vars, etc)
