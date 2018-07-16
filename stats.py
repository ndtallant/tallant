'''
tallant.stats

This file contains functions / helpers for statistical analysis in python.

'''

from statsmodels.stats.multicomp import pairwise_tukeyhsd

def tukey(df, target, factor, sig_lvl=0.05):
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
                               alpha=0.05)
    tukey_.plot_simultaneous()    # Plot group confidence intervals
    plt.vlines(x=df[target].mean(), ymin=-0.5,ymax=100, color="red")

    return tukey_.summary() 
