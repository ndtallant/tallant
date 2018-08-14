'''
Nick Tallant
tallant.pipeline

This file is a machine learning pipeline.
'''
import sys
import logging
import numpy as np
import pandas as pd
from time import strftime as st

sys.path.append('../../')
from stats.stats import MAPE
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.cross_validation import train_test_split

class MagicPipe():
    '''
    This class is a machine learning pipeline.
    Attr: https://github.com/rayidghani/magicloops 
    '''
    def __init__(self, X, y, task, methods, grid, logfile):
        '''
        Inputs
        ------
                X: DataFrame or array, explanatory vars / features/ predictors
                y: Series or array, target
                task: str, 'binary', 'multiclass', or 'regression'
                methods: list, see tallant.machine_learning.grids
                grid: dict, see tallant.machine_learning.grids
                logfile: str, defaults to 'models_<date>' 
        Output
        ------
               DataFrame and log of every model (method, params, evaluation metrics)
        
        ToDo
        ----
               Time-Series holdouts.
               Built in CV.
        ''' 
        self.task = task  
        self.methods = methods
        self.grid = grid 
        self._make_data(X, y)
        self.logfile = '{}_models_{}.log'.format(self.task, st('%m_%d_%Y'))
        self.logger = self._get_logger()
        self.report = self._init_output() 

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
            raise ValueError("Must be array like object.")
        if not isinstance(y, (pd.Series, np.array)):
            raise ValueError("Must be 1-D array like object.")
        if isinstance(X, pd.DataFrame):
            self._features = X.columns
        if self.task == 'multiclass':
            self._labels = pd.Series(y).unique()
        self.X_train, self.X_test, self.y_train,  self.y_test =\
            train_test_split(X, y, random_state=42) 

    def _get_logger(self, lvl=0):
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    def _init_output(self):
        base = ['Method', 'Parameter Set'] 
        if self.task == 'binary':
            metric_cols = ['1%','2%','5%','10%','20%','30%','50%','100%','ROC AUC']
        if self.task == 'multiclass':
            averages = ['micro', 'macro', 'weighted'] 
            metric_cols = [a + '_precision' for a in avg] +\
                          [a + '_roc_auc' for a in avg]
        if self.task == 'regression':
            metric_cols = ['r2_test', 'r2_train', 'MSE_test', 'MSE_train'
                    , 'MAPE_test', 'MAPE_train']
        self._header = base + metric_cols 
        return pd.DataFrame(columns=self._header)

    def _predict(self, method): 
        '''Returns scores and true values in descending order.''' 
        if self.task == 'regression': 
            return method.fit(self.X_train
                    , self.y_train).predict(self.X_test), None
        
        if self.task == 'binary': 
            y_pred = method.fit(self.X_train
                              , self.y_train).predict_proba(self.X_test)[:,1]
        else:
            y_pred = method.fit(self.X_train
                              , self.y_train).predict(self.X_test)
        return zip(*sorted(zip(y_pred, y_test), reverse=True))
    
    def _evaluate(self, y_pred, y_true=None):
        '''
        Using y_true instead of self.y_test for classification
        because they are sorted by the score.
        '''
        if self.task == 'binary': 
            return [self.pct_precision(y_pred, self.y_test, pct) 
                    for pct in [.01, .02, .05, .10, .20, .30, .50, 1]] +\
                    [roc_auc_score(self.y_test, y_pred)]
        
        if self.task == 'multiclass':           
            averages = ['micro', 'macro', 'weighted'] 
            p = [precision_score(y_true, y_pred, labels=self._labels
                , average=avg) for avg in averages]
            ra = [roc_auc_score(y_true, y_pred, labels=self._labels
                , average=avg) for avg in averages]
            return p + ra 

        if self.task == 'regression':
            return [r2_score(self.y_test, y_pred)
                , r2_score(self.y_train, y_pred) 
                , mean_squared_error(self.y_test, y_pred)
                , mean_squared_error(self.y_train, y_pred)
                , MAPE(y_pred, self.y_test) 
                , MAPE(y_pred, self.y_train)] 

    def _output(self, evals):
        out = dict(zip(self._header, evals) 
        self.logger.info(str(out))
        # Add out to the df

    def single_split_loop(self):
        for current in self.methods:
            self.logger.info('Running {}'.format(current))
            try:
                method, params = self.methods[current], self.grid[current]
                for parameter_set in ParameterGrid(params):
                    self.logger.info(parameter_set)
                    method.set_params(**parameter_set)
                    y_pred, y_true = self._predict(method)
                    try:
                        self.logger.debug('Running Metrics') 
                        evals = [method, params] + self._evaluate(y_pred, y_true)
                        self._output(evals)
                    except Exception as e:
                        self.logger.debug('Evaluation Failed with {}'.format(e)) 
            except Exception as e: 
                self.logger.warning('Model failed with {}'.format(e)) 
                continue

    # Binary Task Evaluation Helpers
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
