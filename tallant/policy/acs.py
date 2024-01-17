# American Community Survey
# OUTDATED - UPDATE ME :(
import requests
import numpy as np
import pandas as pd

class ACS:
    def __init__(self, year, dataset, codes, geography, key):
        '''May need to be updated based on the census changes'''
        self.base = ('https://api.census.gov/data/{}/{}'
                     '?get=NAME,{}&for={}&key={}')
        self._code_labels = False
        self.year = year
        self.dataset = dataset
        self.codes = self._parse_codes(codes)
        # lookup mulitple specific geos (not just *)
        self.geography = geography
        self.key = key
        self.query = self.base.format(year
                                    , dataset
                                    , self.codes
                                    , geography
                                    , key) 

    def _parse_codes(self, codes):
        if type(codes) == str:
            return codes
        if type(codes) == list:
            return ','.join(codes) 
        if type(codes) == dict:
            self._code_labels = codes
            return self._parse_codes(list(codes.keys()))
    
    def see_query(self):
        '''Previews ACS query to the screen.'''
        print(self.query.split('key')[0])

    def get_data(self):
        response = requests.get(self.query, timeout=60)
        data = response.json()
        df =  pd.DataFrame(data)
        df.columns = list(df.loc[0]) 
        df.drop(0, inplace=True) 
        if self._code_labels:
            df.rename(columns=self._code_labels, inplace=True)
        df.replace('*****', np.NaN, inplace=True) 
        return df

class newACS:
    def __init__(self, year, dataset, codes, geography, key):
        '''May need to be updated based on the census changes'''
        self.base = ('https://api.census.gov/data/{}/{}/'
                     'profile?get=NAME,{}&for={}&key={}')
