'''
Nick Tallant
tallant.explore

This file contains functions to.

1. Read/Load Data
2. Explore Data
3. Pre-Process and Clean Data
'''

import re



# Cleaning/Processing Section ---------------------------------------

def snakify(txt):
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
    
    s0 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', column_name).title()
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', s0).replace(' ', '')
    s2 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s1)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s2).lower()

#  Exploration Section -----------------------------------------------

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
