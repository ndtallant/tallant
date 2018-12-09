# Machine Learning Method Summaries

## Decision Trees

#### What assumptions does it make about the data?
Decision Trees are used with the assumption that the data is balanced. A training dataset consisting of mostly one class will result in a classifier that is biased towards that majority class.

#### What is it optimizing for?
A decision tree minimizes entropy in the data that it is splitting, which in turn maximizes information gain. The implementation of this model in sci-kit learn can also optimize for "gini" which is functionally equivalent. 

#### What parameters does it have?
- **splitter**: "best" or "random"
- **max\_depth**: integer or None (splits until all leaves are pure or at the min\_samples\_split)
- **min\_samples\_split**: minimum number of samples required to split an internal node
- **max\_features**: number of features considered for each split (defaults to the square root of the number of features)

#### How are those parameters selected?
These parameters are largely influenced by the size of the data. For example a max depth of 8 for a model with 10 features will overfit.

#### How do you score new data and interpret the model's predictions?
Each node of the tree is a logical rule applying to the data point, and a leaf node is the conjunction of the rules of its parents. A data point that meets each logical decision will end up in a leaf node to classify it. The model can also be adjusted for the nodes to show a predicted probability for a positive classification.

#### Implementation Reference
- Low Complexity
- Likely to Overfit
- High Interpretability
- Training Time depends on the depth of the tree and number of features.
- Testing/Scoring Time also depends, but faster than training. 

## Logistic Regression

#### What assumptions does it make about the data?
Logistic Regression assumes there are not many outliers in the data and that the features are linear.

#### What is it optimizing for?
Logistic maximimizes the *log probability* of a positive sample, and minimizes one of two loss functions:
- L1: sum of absolute differences between the true value and the predicted value
- L2: sum of squared differences between the true value and the predicted value

#### What parameters does it have?
- **penalty**: Choose between "L1" and "L2" loss functions.
- **fit\_intercept**: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
- **class\_weight**: Weights associated with classes in the form `{class_label: weight}`.
 
#### How are those parameters selected?
Fitting the intercept, weighting features, and the loss function used all depend on the task at hand.

#### How do you score new data and interpret the model's predictions?
The logistic regression function outputs the predicted probability of a data point being a positive sample. Data with probabilites over the designated threshold are classified as positive, while data with probabilites under the threshold are classified as negative. The threshold is largely influenced by domain knowledge, and is almost never 50%.  

#### Implementation Reference
- Medium Complexity 
- Likely to Overfit
- High Interpretability
- Fast Training Time
- Fast Testing/Scoring Time

## Support Vector Machines

#### What assumptions does it make about the data?
SVMs can handle data with outliers, as the model is trained on only the "support" vectors. The features need to be appropriately scaled.

#### What is it optimizing for?
SVMs optimize on two components:
1. Maximizing the *margin*, the distance between data points closest to the central decision boundary.
2. Minimizing the number of misclassified data points (regularization / hinge loss) 
 
#### What parameters does it have?
- **penalty**: "l1" or "l2"
- **loss**: default is "hinge"
- **class\_weight**: Weights associated with classes in the form `{class_label: weight}`.

#### How are those parameters selected?
The default parameters of sklearn will be used most of the time. Beyond that we want as much information as possible.

#### How do you score new data and interpret the model's predictions?
Classification is rather simple. There is a decision boundary in the vector space separating predicted positives and negatives. Samples below the boundary are positive and samples above are negative. This classification is done with the boundary set at zero. Note the magnitude of the classification prediction is not insightful - for example a sample with -1 is no more likely to be positive than a sample with -0.5. 

#### Implementation Reference
- High Complexity (Depending on the space, particularly high dimensions)
- Less Likely to Overfit
- Low Interpretability
- Slow Training Time
- Slow Testing/Scoring Time
