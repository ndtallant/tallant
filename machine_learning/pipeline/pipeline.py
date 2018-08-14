'''
Nick Tallant
tallant.pipeline

This file is a machine learning pipeline.
'''
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.cross_validation import train_test_split

class MagicPipe(): #maybe inherit from sklearn?    
    '''This class is a machine learning pipeline.'''
    def __init__(self, X, y, task, methods, grid, out_df=True):
        '''DocString!''' 
        # Input Stuff 
        self.task = task  
        self.methods = methods
        self.grid = grid 
        self._make_data(X, y)
        
        # Output Stuff 
        self.logfile 
        self.logger = self._get_logger()
        self.report = self._init_output(self, out_df) 

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, task):
        if task not in ('binary', 'multiclass', 'regression'):
            raise ValueError("Task must be binary, multiclass, or regression.")
        self._task = task 
    
    @property
    def methods(self):
        return self._methods
    
    @methods.setter
    def methods(self, methods):
        if not isinstance(methods, list):
            raise ValueError("Must be a list.")
        self._methods = methods 
    
    @property
    def grid(self):
        return self._X
    
    @grid.setter
    def grid(self, grid):
        if not isinstance(grid, dict):
            raise ValueError("Must be a dictionary.")
        self._grid = grid 
    
    def _make_data(self, X, y):
        '''Makes Train-Test Splits''' 
        if not isinstance(X, (pd.DataFrame, np.array)):
            raise ValueError("Must be array like object")
        if not isinstance(y, (pd.Series, np.array)):
            raise ValueError("Must be 1-D array like object")
        if isinstance(X, pd.DataFrame):
            self._features = X.columns
        
        self.X_train, self.y_train, self.X_test, self.y_test = \
            train_test_split(X, y, random_state=42) 

    def _get_logger(self):
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    def _init_output(self, out_df):
        base = ['Method', 'Parameter Set'] 
        if self.task == 'binary':
            metric_cols = ['1%','2%','5%','10%','20%','30%','50%','100%','ROC AUC']
        if self.task == 'multiclass':
            metric_cols = []
        if self.task == 'regression':
            metric_cols = []
        return pd.DataFrame(columns=base+metric_cols)

    def _predict(self, method): 
        '''Returns scores and true values in descending order.''' 
        if self.task == 'binary': 
            y_pred = method.fit(self.X_train
                              , self.y_train).predict_proba(self.X_test)[:,1]
        else:
            y_pred = method.fit(self.X_train
                              , self.y_train).predict(self.X_test)
        return zip(*sorted(zip(y_pred, y_test), reverse=True))
    
    def _evaluate(self, current, parameter_set):
        if self.task == 'binary': 
            return [self.pct_precision(y_scores, y_true, pct) 
                    for pct in [.01, .02, .05, .10, .20, .30, .50, 1]] +\
                    [roc_auc_score(y_true, y_pred)]
        
        if self.task == 'multiclass':           
            return []

        if self.task == 'regression':
            pass

    def _output(self):
        binary = ['1%','2%','5%','10%','20%','30%','50%','100%','ROC AUC']
        pass
    
    def single_split_loop():
        for current in running:
            print('Running', current)
            try: 
                method, params = self.methods[current], self.grid[current]
                for parameter_set in ParameterGrid(params):
                    # Add logging 
                    print(parameter_set) 
                    method.set_params(**parameter_set)
                    y_scores, y_true = self._predict(method)
                    # Metrics
                    try: 
                        print('helloooo')
                    except ValueError:
                        print('uh oh') 
                    # Output
            except ValueError as e: # All negatives in a split
                print(e) 
                continue
        return rv 

    # Helpers (Binary) 
    @staticmethod
    def pct_precision(y_scores, y_true, pct):
        '''Gets the precision score of a model for a given percentage targeted.'''
        y_scores, y_true = sort_by_score(np.array(y_scores), np.array(y_true))
        preds_at_k = classify_on_threshold(y_scores, pct)
        return precision_score(y_true, preds_at_k)
   
    @staticmethod
    def classify_on_threshold(y_scores, thr):
        '''
        Given sorted prediction scores and a threshold,
        this function classifies each score as positive or negative.
        '''
        positive_bound = int(len(y_scores) * thr)
        return [1 if i < positive_bound else 0 for i in range(len(y_scores))]

    @staticmethod
    def sort_by_score(y_scores, y_true):
        '''
        Sorts scores and true values by scores in descending order.
        https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.argsort.html
        ''' 
        sort_index = np.argsort(y_scores)[::-1]
        return y_scores[sort_index], y_true[sort_index]
