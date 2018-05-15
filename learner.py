'''
Nick Tallant
tallant.learner

This file contains functions for:
- Machine Learning Models/Algorithms
- Feature/Predictor Generation
- Learning Model Evaluation 
'''

import numpy as np
import pandas as pd

# Sci-kit Learn Imports
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy
from sklearn.cross_validation import train_test_split

def rough_binary_eval(x_test, y_test, model):
    '''
    Gives a rough accuracy estimate (don't put faith in this).
    This anser is different than model.score(x,y) - ask why this is. 
    '''

    con_mat = confusion_matrix(y_test, model.predict(x_test))
    return np.trace(con_mat)/np.sum(con_mat)

def k_nearest_nick(x_train, y_train, x_test, y_test):
    '''
    Trains and tests several knn models, 
    provides rough evaluation, and stores information
    in a table.
    
    Input: Training and Testing data from a TT Split
    Output: A dataframe summarizing model performance.
    '''
    list_of_models = [] 
    for k in range(1,11):
        # Maybe add a third loop for Distance metrics later,
        # Euclidean is totally fine for now! 
         
        for weight_func in ['uniform', 'distance']:
            knn = KNeighborsClassifier(n_neighbors=k,
                                       weights=weight_func)
            knn.fit(x_train, y_train)
            
            # maybe score on original data using score method also 
            rough = rough_binary_eval(x_test, y_test, knn) 
            list_of_models.append(['{} neighbors'.format(k),
                                   weight_func,
                                   rough])
            
    return pd.DataFrame(list_of_models, 
                       columns=['Neighbors', 'Weight', 'Performance'])

def compare_trees(x_train, y_train, x_test, y_test):
    depths = [1, 3, 5, 7]
    for d in depths:
        dec_tree = DecisionTreeClassifier(max_depth=d)
        dec_tree.fit(x_train, y_train)
        train_pred = dec_tree.predict(x_train)
        test_pred = dec_tree.predict(x_test)
        # evaluate accuracy
        train_acc = accuracy(train_pred, y_train)
        test_acc = accuracy(test_pred, y_test)
        
        print("Depth: {} | Train acc: {:.2f} \
                | Test acc: {:.2f}".format(d, train_acc, test_acc))

def see_tree_importance(x_train, tree):
    '''Returns a DF of features selected and the models importance'''
    #Find a better way to word that^
    return pd.concat([pd.Series(x_train.columns, name='Feature'),
           pd.Series(tree.feature_importances_, name='Importance')],
               axis=1)
