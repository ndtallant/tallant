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

CLFS = {  'knn': KNeighborsClassifier(n_neighbors=3), 
           'dt': DecisionTreeClassifier(), 
        'logit': LogisticRegression(penalty='l1', C=1e5), 
          'svm': SVC(kernel='linear', probability=True, random_state=0),
           'nb': GaussianNB, 
           'rf': RandomForestClassifier(n_estimators=50, n_jobs=-1), 
          'gbc': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
          'bag': BaggingClassifier(),
          'ada': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)} 

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

def single_split_loop(X_train, y_train, X_test, y_test):
    '''Loops over methods and params using a single train test split''' 
    
    single_results = pd.DataFrame(columns=['Method', 'Parameters'] + EVAL_COLS) 
    
    for current in METHODS:
        print('Running', current)
        method, params = CLFS[current], PARAMS[current]
        for p in ParameterGrid(params):
            print(p) 
            try:
                method.set_params(**p)
                y_pred_probs = method.fit(X_train, y_train).predict_proba(X_test)[:,1]
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                ''' 
                results_df.loc[len(results_df)] = [current, p, roc_auc_score(y_test, y_pred_probs)] + 
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                ''' 
            except IndexError as e:
                print('Error:',e)
                continue
    return single_results

def full_loop(full_data, clean_split=None):
    pass
    for split_set in SPLITS:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    if clean_split:
        clean_split()        

def get_splits(X, y):
    pass
# Cross Validation TTS or Temporal Holdout

'''
for dataset in splits:
    impute missing values
    handle outliers

    for each_method in methods:
        fit on parameter permutation (grid or get_params)
        eval prec, recall, roc_auc, thrs 
        append to results
'''
