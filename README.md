
# Bias-Variance Trade-Off - Lab

## Introduction

In this lab, you'll practice your knowledge on the bias-variance trade-off!

## Objectives

You will be able to: 
- Look at an example where Polynomial regression leads to overfitting
- Understand how bias-variance trade-off relates to underfitting and overfitting

## Let's get started!

We'll try to predict some movie revenues based on certain factors, such as ratings and movie year.


```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('./movie_data_detailed_with_ols.xlsx')
df.head()
```


```python
# Only keep four predictors and transform the with MinMaxScaler

scale = MinMaxScaler()
df = df[[ "domgross", "budget", "imdbRating", "Metascore", "imdbVotes"]]
transformed = scale.fit_transform(df)
pd_df = pd.DataFrame(transformed, columns = df.columns)
pd_df.head()
```

## Split the data into a test and train set


```python
# domgross is the outcome variable
```


```python
#Your code here
```

## Fit a regression model to the training data and look at the coefficients


```python
#Your code 
```

## Plot the training predictions against the actual data (y_hat_train vs. y_train)

Let's plot our result for the train data. Because we have multiple predictors, we can not simply plot the income variable X on the x-axis and target y on the y-axis. Lets plot 
- a line showing the diagonal of y_train. The actual y_train values are on this line
- next, make a scatter plot that takes the actual y_train on the x-axis and the predictions using the model on the y-axis. You will see points scattered around the line. The horizontal distances between the points and the lines are the errors.


```python
import matplotlib.pyplot as plt
%matplotlib inline
# your code here
```

## Plot the test predictions against the actual data (y_hat_test vs. y_test)

Do the same thing for the test data.


```python
# your code here
```

## Calculate the bias
Write a formula to calculate the bias of a models predictions given the actual data: $Bias(\hat{f}(x)) = E[\hat{f}(x)-f(x)]$   
(The expected value can simply be taken as the mean or average value.)  



```python
import numpy as np
def bias(y, y_hat):
    pass
```

## Calculate the variance
Write a formula to calculate the variance of a model's predictions: $Var(\hat{f}(x)) = E[\hat{f}(x)^2] - \big(E[\hat{f}(x)]\big)^2$


```python
def variance(y_hat):
    pass
```

## Use your functions to calculate the bias and variance of your model. Do this seperately for the train and test sets.


```python
# code for train set bias and variance
```


```python
# code for test set bias and variance
```

## Describe in words what these numbers can tell you.

Your description here (this cell is formatted using markdown)

## Overfit a new model by creating additional features by raising current features to various powers.

Use `PolynomialFeatures` with degree 3. 

**Important note:** By including this, you don't only take polynomials of single variables, but you also combine variables, eg:

$ \text{Budget} * \text{MetaScore} ^ 2 $

What you're essentially doing is taking interactions and creating polynomials at the same time! Have a look at how many columns we get using `np.shape`. Quite a few!



```python
from sklearn.preprocessing import PolynomialFeatures\
# your code here
```

## Plot your overfitted model's training predictions against the actual data


```python
# your code here
```

Wow, we almost get a perfect fit!

## Calculate the bias and variance for the train set


```python
# your code here
```

## Plot your overfitted model's test predictions against the actual data.


```python
# your code here
```

##  Calculate the bias and variance for the train set.


```python
# your code here
```

## Describe what you notice about the bias and variance statistics for your overfit model

The bias and variance for the test set both increased drastically in the overfit model.

## Level Up - Optional

In this lab we went from 4 predictors to 35 by adding polynomials and interactions, using `PolynomialFeatures`. That being said, where 35 leads to overfitting, there are probably ways to improve by just adding a few polynomials. Feel free to experiment and see how bias and variance improve!

## Summary

This lab gave you insight in how bias and variance change for a training and test set by using a pretty "simple" model, and a very complex model. 
