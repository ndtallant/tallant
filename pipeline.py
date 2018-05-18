'''
Nick Tallant
tallant.pipeline

This file is a machine learning pipeline.
'''

import numpy as np
import pandas as pd

# Machine Learning Methods 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 

#keep in mind what weak learners you want
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 

# Evaluation Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Cross Validation
from sklearn.model_selection import ParameterGrid
from sklearn.cross_validation import train_test_split

THRESHOLDS = [.01, .02, .05, .10, .20, .30, .50]
EVAL_COLS = ['1%', '2%', '5%', '10%', '20%', '30%', '50%', 'ROC AUC']
METHODS = ['knn', 'dt', 'logit', 'svm', 'rf', 'gbc', 'bag', 'ada'] 
BASICS = ['knn', 'dt', 'logit']

CLFS = {  'knn': KNeighborsClassifier(n_neighbors=3), 
           'dt': DecisionTreeClassifier(), 
        'logit': LogisticRegression(penalty='l1', C=1e5), 
          'svm': SVC(kernel='linear', probability=True, random_state=0),
           'nb': GaussianNB, 
           'rf': RandomForestClassifier(n_estimators=50, n_jobs=-1), 
          'gbc': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, 
                                            max_depth=6, n_estimators=10),
          'bag': BaggingClassifier(),
          'ada': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
                                    algorithm="SAMME", n_estimators=200)} 

PARAMS = {'knn': {'n_neighbors': [1, 5, 10, 25, 50, 100],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree']},
           
          'dt': {'criterion': ['gini', 'entropy'], 
                 'max_depth': [1, 5, 10, 20, 50, 100],
                 'min_samples_split': [2, 5, 10]},
    
          'logit': {'penalty': ['l1', 'l2'], 'C': [10**n for n in range(-4,2)]},
    
          'svm': {'C' :[10**n for n in range(-5,2)], 'kernel': ['linear']},
    
          'nb': {},
    
          'rf': {'n_estimators': [10, 100], 
                 'max_depth': [5, 50], 
                 'max_features': ['sqrt', 'log2'],
                 'min_samples_split': [2, 10], 
                 'n_jobs': [-1]},
    
          'gbc': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    
          'bag': {}, # find params you like 
    
          'ada': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10**n for n in range(5)]}}

SMALL = {'knn': {'n_neighbors': [5, 10, 25],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto']},
           
         'dt': {'criterion': ['gini', 'entropy'], 
                 'max_depth': [1, 5, 10],
                 'min_samples_split': [2, 5]},

         'logit': {'penalty': ['l2'], 'C': [10**n for n in range(-2,2)]}}

def single_split_loop(X_train, y_train, X_test, y_test, quick=False):
    '''Loops over methods and params using a single train test split''' 
     
    rv = pd.DataFrame(columns=['Method', 'Parameters'] + EVAL_COLS) 
    running = BASICS if quick else METHODS 
    param_gr = SMALL if quick else PARAMS 
    
    for current in running:
        print('Running', current)
        try: 
            method, params = CLFS[current], param_gr[current]
            for p in ParameterGrid(params):
                print(p) 
                method.set_params(**p)
                y_pred_probs = method.fit(X_train, y_train).predict_proba(X_test)[:,1]
                y_scores, y_true = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    
                try: # Can have issues with multiclass tasks 
                    roc_auc = roc_auc_score(y_true, y_scores)
                except ValueError:
                    roc_auc = 'N/A'
                    
                front = [current, p, roc_auc] 
                back = [thr_precision(y_scores, y_true, thr) for thr in THRESHOLDS]
                rv.loc[len(rv)] = front + back 
        except ValueError: # All negatives in a split
            continue
    return rv 

def thr_precision(y_scores, y_true, thr):
    '''Gets the precision score of a model for a given threshold.'''
    y_scores, y_true = sort_by_score(np.array(y_scores), np.array(y_true))
    preds_at_k = classify_on_threshold(y_scores, thr)
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
