'''
Nick Tallant
tallant.utils

This file contains misc. helper functions
'''
import re
import json
import pandas as pd

def snakify(feature, verbose=False):
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

def make_json(dictionary, filename):
    '''Makes a json from a dictionary'''
    data = json.dumps(dictionary)
    f = open(filename)
    f.write(data)
    f.close()

def read_json(filename):
    '''Makes a dictionary from a json'''
    with open(filename) as f:
        data = f.read()
    return json.loads(data)

def read_sql(filename, con):
    '''Makes a dataframe from a sql file'''
    with open(filename) as f:
            sql = f.read()
    return pd.read_sql(sql, con)
