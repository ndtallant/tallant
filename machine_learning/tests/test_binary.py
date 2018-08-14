from sklearn.datasets import load_breast_cancer
import sys
sys.path.append('../')
from pipeline.pipeline import MagicPipe

data = load_breast_cancer()
X, y = data.data, data.target
print(type(X), type(y))
binary_pipe = MagicPipe(X=X, y=y, task='binary', methods=[]
        , grid={}, logfile='')

