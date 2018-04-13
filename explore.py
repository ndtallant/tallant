'''
Nick Tallant
tallant.explore

This file contains functions to.

1. Read/Load Data
2. Explore Data
3. Pre-Process and Clean Data
'''
import re
import pandas as pd
import numpy as np
from warnings import warn
import plotly.plotly as py
import plotly.graph_objs as go

def load_data(file_name, file_type='csv', verbose=True, meta_data=None):
    '''
    Loads in data to a dataframe, cleans columns,
    and provides basic summary information.
    More info can be provided if given a data dict.
    
    Input: meta_data - takes a data dictionary as a csv 
                       with three columns: feature, dtype, dtype2.
    '''
    
    if file_type == 'csv':
        df = pd.read_csv(file_name)

    snakify_cols(df, verbose=verbose)
    
    if meta_data:
        # do something with each feature based on dtype
        # stock pandas dtype isn't enough
        # but it might be for now .....
        pass
    else:    
        if verbose: 
            warn('Meta data is useful! Check docstring.')
    return df

def explore_categorical(df, features):
    '''
    Explores one or more categorical features from a list
    '''
    for feature in features:
        print(df[feature].value_counts())



# Helpers -----------------------------------------------------------


def snakify_cols(df, verbose=True):
    '''
    Takes in a dataframe and alters the columns/features
    to snake_case inplace.
    '''

    cols = {col:snakify(col, verbose=verbose) for col in df.columns}
    df.rename(columns=cols, inplace=True)

def snakify(feature, verbose=True):
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
        warn(f'Rename {snake_feature}')
    # find a way to get rid of source line for this warning. 
    return snake_feature


#df.dtypes just returns object, int, or float, not str!
# this might not be worth it though....
def see_feature_types(df, via_plotly=True):
    '''
    This function takes in a DataFrame and returns
    a plotly table (which can be rendered in jupyter OR
    seen on your plotly account - or will just return a dictionary
    if via_plotly is set to False. 
    '''
    trace = go.Table(
            header={'values':['Feature', 'Type', 'Example']},
            cells={'values':[[c, df[c].loc[0]]
                               for c in df.columns]})
    data = [trace]
    py.iplot(data, filename='feature_table')

def time_plots(df):
    '''
    Plots the frequency of a observations by month and weekday.
    
    Input: df (pandas DataFrame w/ date column)
    Output: plots of observations by month and weekday.
    '''
    pass

    # clip month?
    # make column datetime objects
    # make month feature
    # make weekday feature
    # plot!
