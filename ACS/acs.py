# American Community Survey

class ACS:
    def __init__(self, year, dataset, codes, geography, key):
        self.base = 'https://api.census.gov/data/{}/{}?get=NAME,{}&for={}&key={}'
        self.year = year
        self.dataset = dataset
        self.codes = ','.join(codes)
        self.geography = geography # lookup mulitple specific geos (not just *)
        self.key = key
        self.query = self.base.format(year, dataset, codes, geography, key) 

    def see_query(self):
        '''Previews ACS query to the screen.'''
        print(self.query[:len(key)])
