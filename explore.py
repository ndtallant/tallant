'''
Nickallant
tallant.explore

Thisile contains functions to.

1.ead/Load Data
2.xplore Data
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

class Prometheus:
    '''Loads, cleans, and exports data'''
    
    def __init__(self, data=None, verbose=True):
        self.df = self.load_data(data) 
        self.snakify_cols()
        self.numeric_features = [] 
        self.categorical_features = [] 
        self.get_feature_types()
        for feature in self.df.columns:
            print(feature, self.nan_scan(feature))

    def load_data(self, data, verbose=True):
        '''Loads in data to a dataframe.'''
        if isinstance(data, pd.DataFrame): 
            return data 
        elif isinstance(data, str):
            if data.endswith('csv'):
                return pd.read_csv(data)
            elif data.endswith('json'):
                return pd.read_json(data)
        raise ValueError('I need a csv, json, or DataFrame!')
        
    def snakify_cols(self, verbose=True):
        '''
        Takes in a dataframe and alters the columns/features
        to snake_case inplace.
        '''
        cols = {col:snakify(col, verbose=verbose) for col in self.df.columns}
        self.df.rename(columns=cols, inplace=True)

    def get_var_category(self, feature):
        '''Returns either Numerical, Date, Text, or Categorical'''
        unique_count = self.df[feature].nunique(dropna=False)
        total_count = len(self.df[feature])
        if pd.api.types.is_numeric_dtype(self.df[feature]):
            return 'Numerical'
        elif pd.api.types.is_datetime64_dtype(self.df[feature]):
            return 'Date'
        elif unique_count == total_count:
            return 'Text (Unique)'
        else:
            return 'Categorical'

    def get_feature_types(self):
        for feature in self.df.columns:
            cat = self.get_var_category(feature)
            if cat == 'Numerical':
                self.numeric_features.append(feature)
            elif cat == 'Categorical':
                self.categorical_features.append(feature)

    # Look into Imputer classes from sk-learn
    def replace_na_random(self, feature, lower, upper):
        '''
        Replaces any Null values in a feature with a random int
        between the lower and upper bounds
        '''
        pass

    def nan_scan(self, feature):
        if self.df[feature].isnull().values.any():
            return 'Has Nans' 
        return 'All Good'

    def bound_feature(self, feature):
        '''why did you make this?''' 
        foi = df[feature]
        ok_vals = self.df[abs(df[feature]) < 1][feature]
        foi = foi.where(abs(foi) < 1, ok_vals.mean())
        self.df[feature] = foi 

#ev to-do
#dealingith outliers, 
#discretizingontinuous/categorical data


#plotistograms/kde plots of distributions, 
#scatterlot to show relationship between two vars, etc)
