'''
Nick Tallant
tallant.explore

This file contains functions to.

1. Read/Load Data
2. Explore Data
3. Pre-Process and Clean Data
'''
import re
import random
import pandas as pd
import numpy as np
from warnings import warn
#import plotly.plotly as py
#import plotly.graph_objs as go

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

# Look into Imputer classes from sk-learn
def replace_na_random(df, feature, lower, upper):
    '''
    Replaces any Null values in a feature with a random int
    between the lower and upper bounds
    '''
    for index, row in df.iterrows():
        if pd.isna(row[feature]):
            df.loc[index, feature] = random.randint(lower, upper)

def nan_scan(df):
    for feature in df.columns:
        if df[feature].isnull().values.any():
            message = 'Has Nans' 
        else:
            message = 'All Good'
        print(message, "-->", feature)

def bound_feature(df, feature):
    foi = df[feature]
    ok_vals = df[abs(df[feature]) < 1][feature]
    foi = foi.where(abs(foi) < 1, ok_vals.mean())
    df[feature] = foi 

class FatherTime:
    '''
    Class used to explore and create temporal features.
    
    https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    '''
    def __init__(self, feature):
        '''
        Just takes in a string of the feature of interest.
        Does not wrap an entire dataframe for memory reasons.
        '''
        self.feature = feature 

    def make_datetime(self, df, feature):
        '''
        Creates a datetime series from a feature, and makes
        the self.feature attr the new datetime series for further 
        exploration and feature generation.
        '''
        # Think about exception handling 
        df[feature] = pd.to_datetime(df[feature]) 
        
    def get_months(self, df, feature):
        '''
        Returns the name of the month from timeseries data.
        Can return just the series or will add the feature to the df.
        
        Uses datetime objects (use make_datetime method). 
        ''' 
        month_map = { 1: 'January',
                      2: 'February',
                      3: 'March',
                      4: 'April',
                      5: 'May',
                      6: 'June',
                      7: 'July',
                      8: 'August',
                      9: 'September',
                     10: 'October',
                     11: 'November',
                     12: 'December'}

        if humanize: 
            df['month'] = df[feature].apply(lambda x: month_map[x.month])
        else:
            df['month'] = df[feature].apply(lambda x: x.month)
    
    def get_weekday(self, df, feature, humanize=True):
        '''
        Returns the name of the weekday from timeseries data.
        Can return just the series or will add the feature to the df.
        
        Uses datetime objects (use make_datetime method). 
        '''
        day_map = {0: 'Monday',
                   1: 'Tuesday',
                   2: 'Wednesday',
                   3: 'Thursday',
                   4: 'Friday',
                   5: 'Saturday',
                   6: 'Sunday'}

        if humanize: 
            df['weekday'] = df[feature].apply(lambda x: day_map[x.weekday()])
        else:
            df['weekday'] = df[feature].apply(lambda x: x.weekday())

    def get_day_of_year(self, df, feature):
        '''
        Returns the day of year as an int from timeseries data.
        Can return just the series or will add the feature to the df.
        '''
        df['day_of_year'] = df[feature].apply(lambda x: int(x.strftime('%j')))

    def get_military_hour(self, df, feature):
        '''
        Returns the hour from 0 to 23 from timeseries data.
        Can return just the series or will add the feature to the df.
        '''
        df['hour'] = df[feature].apply(lambda x: int(x.strftime('%H')))

    def get_radial_hour(self, df, feature):
        '''
        Returns the time of day as the position on a clock from timeseries data.
        This allows 11 p.m. to be as close to 1 a.m. as 9 p.m. for distance. 
        Can return just the series or will add the feature to the df.
        '''
        df['sin_hour'] = np.sin(np.pi*df[feature]/12)
        df['cos_hour'] = np.cos(np.pi*df[feature]/12)
  
    def plot_clock(self, df, feature): 
        '''
        Plots the hours of a dataframe in 2d space like a clock.
        '''
        df.plot.scatter('sin_hour','cos_hour').set_aspect('equal')

    def get_season(self, df, feature):
        '''
        Returns the season from timeseries data.
        Can return just the series or will add the feature to the df.
        '''
        #Going to need to define custom bins 
        #pd.cut(df['day_of_year'], ['Spring', 'Summer', 'Fall', 'Winter']) 
        raise NotImplementedError

    def get_radial_season(self, df, feature):
        '''
        Returns the day of the year as a position on a circle to 
        explore distance.
        Can return just the series or will add the feature to the df.
        '''
        df['sin_day'] = np.sin(2*np.pi*df['day_of_year']/365)
        df['cos_day'] = np.cos(2*np.pi*df['day_of_year']/365)
    
    def season_circle(self, df, feature): 
        '''
        Plots the days of a dataframe in 2d space as a circle.
        '''
        df.plot.scatter('sin_day','cos_day').set_aspect('equal')
   
    def time_plots(self):
        '''
        Plots the frequency of a observations by month and weekday.
       
        Input: df (pandas DataFrame w/ date column)
        Output: plots of observations by month and weekday.
        '''
        raise NotImplementedError 

class Atlas:
    '''
    Class used to explore and create geospatial features.
    '''
    pass 

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
        #warn(f'Rename {snake_feature}') 3.6 is the best
        warn('Rename {}'.format(snake_feature))
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


# Dev to-do
#dealing with outliers, 
#discretizing continuous/categorical data
#plot histograms/kde plots of distributions, 
#scatter plots to show relationship between two vars, etc)
