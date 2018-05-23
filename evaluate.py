'''
Nick Tallant
tallant.evaluate

This file contains functions for Machine Learning Model Evaluation 
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Methods 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 

# Evaluation Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, classification_report, confusion_matrix

EVAL_COLS = ['1%', '2%', '5%', '10%', '20%', '30%', '50%', 'ROC AUC']

def get_top_models(results_file):
    '''Returns a DataFrame of the top models from the results csv.'''
    rv = pd.DataFrame() 
    df = pd.read_csv(results_file)
    for level in EVAL_COLS:
        result = df[df[level] == df[level].max()]
        if result[level].mean() != 1:
            rv = rv.append(result)
    return rv 

def plot_roc(name, y_scores, y_true, output_type):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc='lower right')
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

def thresh_metric(y_scores, y_true, thr, recall=False):
    '''
    Gets the precision score of a model for a given threshold.
    '''
    y_scores, y_true = sort_by_score(np.array(y_scores), np.array(y_true))
    preds_at_k = classify_on_threshold(y_scores, thr)
    if recall:
        return recall_score(y_true, preds_at_k)
    return precision_score(y_true, preds_at_k)

def classify_on_threshold(y_scores, thr):
    '''
    Given sorted prediction scores and a threshold,
    this function classifies each score as positive or negative.
    '''
    positive_bound = int(len(y_scores) * thr)
    return [1 if i < positive_bound else 0 for i in range(len(y_scores))]

def sort_by_score(y_scores, y_true):
    '''
    Sorts scores and true values by scores in descending order.
    https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.argsort.html
    ''' 
    sort_index = np.argsort(y_scores)[::-1]
    return y_scores[sort_index], y_true[sort_index]

def plot_precision_recall_n(y_true, y_score, model_name, save=True):

    p_curve, r_curve, pr_thr = precision_recall_curve(y_true, y_score)
    p_curve = p_curve[:-1]
    r_curve = r_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    plt.title(model_name)
    if save:
        outfile = 'prec_recall_{}'.fomat(model_name) 
        plt.savefig(outfile)
    else:
        plt.show()
