'''
This file holds functions for doing useful things
with dates and times.

https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

The bottom half of this file is broken!!!!!
'''

import datetime
import numpy as np
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
    start = center - pd.Timedelta(margin)
    end = center + pd.Timedelta(margin)
    mask = (df.index > start) & (df.index <= end)
    return df[mask]


def get_radial_hour():
    '''
    Returns the time of day as the position on a clock from timeseries data.
    This allows 11 p.m. to be as close to 1 a.m. as 9 p.m. for distance.
    Can return just the series or will add the feature to the df.
    '''
    df['sin_hour'] = np.sin(np.pi*df[feature]/12)
    df['cos_hour'] = np.cos(np.pi*df[feature]/12)

def get_season():
    '''
    Returns the season from timeseries data.
    Can return just the series or will add the feature to the df.
    '''
    df['season'] = df.day_of_year.apply(_season_helper)

def _season_helper(df, day_of_year):
    '''Helps create seasons'''
    if day_of_year < 79 or day_of_year > 344:
        return 'Winter'
    if day_of_year > 264:
        return 'Autumn'
    if day_of_year > 171:
        return 'Summer'
    return 'Spring'

def get_radial_season(df):
    '''
    Returns the day of the year as a position on a circle to
    explore distance.
    Can return just the series or will add the feature to the df.
    '''
    df['sin_day'] = np.sin(2*np.pi*df['day_of_year']/365)
    df['cos_day'] = np.cos(2*np.pi*df['day_of_year']/365)
