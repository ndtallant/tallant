# American Community Survey
import requests
import pandas as pd

class ACS:
    def __init__(self, year, dataset, codes, geography, key):
        '''May need to be updated based on the census changes'''
        self.base = ('https://api.census.gov/data/{}/{}/'
                     'profile?get=NAME,{}&for={}&key={}')
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
            return self._parse_codes(self, codes.keys())
    
    def see_query(self):
        '''Previews ACS query to the screen.'''
        print(self.query.split('key')[0])

    def get_data(self):
        response = requests.get(self.query)
        data = response.json()
        df =  pd.read_json(data)
        df.columns = list(df.loc[0]) 
        df.drop(0, inplace=True) 
        if self._code_labels:
            df.rename(colums=_code_labels, inplace=True)
        df.replace('*****', np.NaN, inplace=True) 
        return df
