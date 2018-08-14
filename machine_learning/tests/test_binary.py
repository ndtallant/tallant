from sklearn.datasets import load_breast_cancer
import sys
sys.path.append('../')
from pipeline import MagicPipe
from grids import test_clf, short_clf

data = load_breast_cancer()
X, y = data.data, data.target

binary_pipe = MagicPipe(X=X, y=y, task='binary', method_list=test_clf
        , grid=short_clf, logfile='')

binary_pipe.single_split_loop()

print(binary_pipe.report)
