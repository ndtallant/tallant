'''
This file holds functions for doing useful things 
with dates and times.

'''

import pandas as pd

def get_time_window(df, center, margin=''):
    '''
    Returns a chunk of a dataframe with the minimum
    DateTime as center - margin and maxiumum center + margin.

    Input: df: DataFrame with DateTime index.
           center: Timestamp like object.
           margin: String in time delta format.
    
    Output: DataFrame with DateTime index.

    Examples:

    '''
    if not margin:
        raise ValueError('Needs a margin with Timedelta formatting')
    center = pd.to_datetime(center) 
    start = center - pd.Timedelta(margin)
    end = center + pd.Timedelta(margin)
    mask = (df.index > start) & (df.index <= end)
    return df[mask] 
