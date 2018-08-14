'''
Nick Tallant
tallant.grids

This file is a collection of grids for machine learning.
Specifically different groups of methods and their parameters
for automatic hyperparameter tuning. Run these grids in the
appropriate pipelines.
'''

# Classifiers -------------------------------------------------------

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 

# Regressors --------------------------------------------------------
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.ensemble import BaggingRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 

test_clf = ['dt', 'logit'] 
test_reg = ['lm', 'lasso']
ensembles_only = ['rf', 'gb', 'bag', 'ada']

# All sklearn classifiers are multiclass out of the box. 
classifiers = {'knn': KNeighborsClassifier(), 
                'dt': DecisionTreeClassifier(), 
             'logit': LogisticRegression(), 
               'svm': SVC(),
                'nb': GaussianNB(), 
                'rf': RandomForestClassifier(), 
                'gb': GradientBoostingClassifier(),
               'bag': BaggingClassifier(),
               'ada': AdaBoostClassifier(), 
} 

regressors = {'lm': LinearRegression(),
            'lars': Lars(),
           'lasso': Lasso(),                             
           'ridge': Ridge(),
         'elastic': ElasticNet(),
           'theil': TheilSenRegressor(),
         'passive': PassiveAggressiveRegressor(),
             'bag': BaggingRegressor(), 
             'ada': AdaBoostRegressor(),
              'rf': RandomForestRegressor(),
              'gb': GradientBoostingRegressor(), 
         }

# Parameters
long_clf = {'knn': {'n_neighbors': [1, 5, 10, 25, 50, 100],
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

short_clf = {'knn': {'n_neighbors': [5, 10, 25],
                     'weights': ['uniform', 'distance'],
                     'algorithm': ['auto']},

         'dt': {'criterion': ['gini', 'entropy'], 
                 'max_depth': [1, 5, 10],
                 'min_samples_split': [2, 5]},
         
         'logit': {'penalty': ['l2'], 'C': [10**n for n in range(-2,2)]}}
