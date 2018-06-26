'''
Nick Tallant
tallant.feature

This file contains classes for feature generation.

- Kronos: Temporal feature generation
- Atlas:  Geospatial feature generation

'''
import pandas as pd
import numpy as np
import datetime

class Kronos:
    '''
    Class used to explore and create temporal features.
    Kronos was the king of the Titans and the god of Time.

    https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    '''
    def __init__(self, df, feature, create_date=False, humanize=False):
        '''
        Just takes in a string of the feature of interest.
        Does not wrap an entire dataframe for memory reasons.
        '''
        self.df = df 
        self.feature = feature
        if create_date:
            self.create_date_features(humanize=humanize)

    def create_date_features(self, humanize=False):
        self.make_datetime()
        self.get_month(humanize=humanize)
        self.get_year() 
        self.get_weekday(humanize=humanize)
        self.get_day_of_year()
        self.get_radial_season()
        self.get_year() 
        self.get_season()
    
    def make_datetime(self):
        '''
        Creates a datetime series from a feature, and makes
        the self.feature attr the new datetime series for further 
        exploration and feature generation.
        '''
        # Think about exception handling 
        self.df[self.feature] = pd.to_datetime(self.df[self.feature]) 
        
    def get_year(self):
        self.df['year'] = self.df[self.feature].apply(lambda x: x.year)
    
    def get_month(self, humanize=False):
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
            self.df['month'] = self.df[self.feature].apply(lambda x: month_map[x.month])
        else:
            self.df['month'] = self.df[self.feature].apply(lambda x: x.month)
    
    def get_weekday(self, humanize=False):
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
            self.df['weekday'] = self.df[self.feature].apply(lambda x: day_map[x.weekday()])
        else:
            self.df['weekday'] = self.df[self.feature].apply(lambda x: x.weekday())

    def get_day_of_year(self):
        '''
        Returns the day of year as an int from timeseries data.
        Can return just the series or will add the feature to the df.
        '''
        self.df['day_of_year'] = self.df[self.feature].apply(lambda x: int(x.strftime('%j')))

    def get_military_hour(self):
        '''
        Returns the hour from 0 to 23 from timeseries data.
        Can return just the series or will add the feature to the df.
        '''
        self.df['hour'] = self.df[self.feature].apply(lambda x: int(x.strftime('%H')))

    def get_radial_hour(self):
        '''
        Returns the time of day as the position on a clock from timeseries data.
        This allows 11 p.m. to be as close to 1 a.m. as 9 p.m. for distance. 
        Can return just the series or will add the feature to the df.
        '''
        self.df['sin_hour'] = np.sin(np.pi*self.df[self.feature]/12)
        self.df['cos_hour'] = np.cos(np.pi*self.df[self.feature]/12)
  
    def plot_clock(self): 
        '''
        Plots the hours of a dataframe in 2d space like a clock.
        '''
        self.df.plot.scatter('sin_hour','cos_hour').set_aspect('equal')

    def get_season(self):
        '''
        Returns the season from timeseries data.
        Can return just the series or will add the feature to the df.
        '''
        self.df['season'] = self.df.day_of_year.apply(self._season_helper) 
        
    def _season_helper(self, day_of_year):
        '''Helps create seasons'''
        if day_of_year < 79 or day_of_year > 344:
            return 'Winter'
        if day_of_year > 264:
            return 'Autumn'
        if day_of_year > 171:
            return 'Summer'
        return 'Spring'

    def get_radial_season(self):
        '''
        Returns the day of the year as a position on a circle to 
        explore distance.
        Can return just the series or will add the feature to the df.
        '''
        self.df['sin_day'] = np.sin(2*np.pi*self.df['day_of_year']/365)
        self.df['cos_day'] = np.cos(2*np.pi*self.df['day_of_year']/365)
    
    def season_circle(self, target=None): 
        '''
        Plots the days of a dataframe in 2d space as a circle.
        '''
        self.df.plot.scatter('sin_day','cos_day').set_aspect('equal')
   
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
