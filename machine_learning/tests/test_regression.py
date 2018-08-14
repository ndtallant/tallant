from sklearn.datasets import load_boston
import sys
sys.path.append('../')
from pipeline import MagicPipe
from grids import test_reg, short_reg

data = load_boston()
X, y = data.data, data.target

pipe = MagicPipe(X=X, y=y, task='regression', method_list=test_reg
        , grid=short_reg, logfile='')

pipe.single_split_loop()

print(pipe.report)
