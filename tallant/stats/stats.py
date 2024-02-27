'''
tallant.stats

This file contains functions / helpers for statistical analysis in python.

'''
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats as scistat
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tools.eval_measures import mse, rmse, bias

def correlation_summary(df, columns:list
        , hm_kwargs=None):
    '''
    Returns a seaborn heatmap (fig) of correlations for all given
    columns and a dataframe with results of pearson tests between
    every combination of 2 columns.
    '''
    hm_kwargs = {'vmin':0, 'vmax':1, 'annot':True} if hm_kwargs is None else hm_kwargs
    corrs = df[columns].corr()
    heatmap = sns.heatmap(corrs, **hm_kwargs ) 
    l = [] 
    for a, b in set(combinations(columns, 2)):    
        r, p = scistat.pearsonr(df[a], df[b])        
        l.append({'feature1': a
            , 'feature2': b
            , 'r': r
            , 'p-value': p})
    return heatmap, pd.DataFrame(l) 

def tukey_summary(df, target, factor, sig_lvl=0.05):
    '''
    For deciding what levels within a factor constitute 
    significantly distinct groups. This function tests every
    combination of levels within the factor, plots the confidence intervals,
    and gives a summary report where True means they are distinct.

    Inputs: df - DataFrame
            target - the endogenoues variable / variable of interest
            factor - the categorical variable you are looking to analyze
            sig_lvl - the significance level for the Tukey test.

    Outputs: A plot, and a summary (statsmodels.simpletable object)
    '''
    tukey_ = pairwise_tukeyhsd(endog=df[target],    
                               groups=df[factor], 
                               alpha=sig_lvl)
    tukey_.plot_simultaneous() # Plot group confidence intervals
    plt.axvline(x=df[target].mean(), color='r')

    return tukey_.summary()

def OLS(df, y, X=None):
    '''I don't like the statsmodels api so here's this.''' 
    X = df[X] if X else df.drop(y, axis=1)
    X = add_constant(X)
    y = df[y]
    return sm.OLS(np.asarray(y), np.asarray(X)).fit()

def make_restricted(df, exog, endog):
    data = df[exog + [endog]]
    data = xp.dumb_cats(data, drop=True)
    return OLS(data, endog)

def exclusion_test(unrestricted, restricted, sig_lvl=0.05):
    '''Returns True if significant difference between unrestricted and restricted model.'''
    dfn = restricted.df_resid - unrestricted.df_resid
    numer = (unrestricted.rsquared - restricted.rsquared) / dfn
    denom = (1 - unrestricted.rsquared) / unrestricted.df_resid
    F = numer / denom
    crit_val = f_dist.ppf(q=1-sig_lvl, dfn=dfn, dfd=unrestricted.df_resid)
    return F > crit_val

def MAPE(y_pred, y_true):
    '''
    Mean Absolute Percentage Error
    
    - It cannot be used if there are zero values.
    - Models with high predictions have no upper limit to the percentage error.
    - When MAPE is used to compare the accuracy of prediction methods, it is 
      biased towards models whose predictions are too low.
    '''
    pred = np.asarray(y_pred)
    true = np.asarray(y_true)
    return round(np.sum(np.fabs((pred - true)/pred))/len(true) * 100, 3)

def MSE(y_pred, y_true):
    return mse(y_pred, y_true)

def RMSE(y_pred, y_true):
    return rmse(y_pred, y_true)

def Bias(y_pred, y_true):
    return bias(y_pred, y_true)
