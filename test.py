from pipeline import single_split_loop 
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

data = load_iris()

X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print(single_split_loop(X_train, y_train, X_test, y_test))
