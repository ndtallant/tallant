import re
import pandas as pd
from warnings import warn

def quick_summary(df):
    '''Shows each column, if it has nans, its type, and an example value'''
    rv = pd.concat([df.isna().sum() / len(df),
                      df.apply(get_column_category),
                      df.nunique(),
                      df.loc[0]], axis=1)
    rv.columns = ['% Missing', 'Type', 'Uniques', 'Example']
    return rv

def get_column_category(feature):
    '''Returns either Numerical, Date, Text, or Categorical'''

    if feature.nunique() == 2:
        return 'Binary'
    if feature.dtype == 'O':
        return 'Categorical'
    if pd.api.types.is_numeric_dtype(feature):
        return 'Numeric'
    if pd.api.types.is_datetime64_dtype(feature):
        return 'Date'
    if feature.nunique(dropna=False) == len(feature):
        return 'Text (Unique)'

def dumb_cats(df, drop=False, sparse=False):
    '''Dummifies all categorical features (dtype object)'''
    dumb_df = pd.DataFrame(index=df.index)
    for feature in df.columns:
        if df[feature].dtype == 'O':
            dummied = pd.get_dummies(df[feature], prefix=feature, drop_first=drop, sparse=sparse)
            dumb_df = dumb_df.join(dummied)
        else:
            dumb_df = dumb_df.join(df[feature])
    return dumb_df

def flip_a_bin(df, feature):
    '''Reverses the binary coding on a feature'''
    try:
        pos, neg = df[feature].unique()
    except ValueError:
        print('Not binary')
    df[feature] = df[feature].apply(lambda x: pos if x == neg else neg)

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
        warn('Rename {}'.format(snake_feature))
    return snake_feature
