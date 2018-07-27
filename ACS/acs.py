# American Community Survey
import requests
import pandas as pd

class ACS:
    def __init__(self, year, dataset, codes, geography, key):
        self.base = 'https://api.census.gov/data/{}/{}/profile?get=NAME,{}&for={}&key={}'
        self.year = year
        self.dataset = dataset
        self.codes = ','.join(codes)
        # lookup mulitple specific geos (not just *)
        self.geography = geography
        self.key = key
        self.query = self.base.format(year, dataset
                                    , self.codes
                                    , geography
                                    , key) 

    def see_query(self):
        '''Previews ACS query to the screen.'''
        print(self.query.split('key')[0])

    def get_data(self):
        pass
