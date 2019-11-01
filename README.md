
# Bias-Variance Tradeoff - Lab

## Introduction

In this lab, you'll practice the concepts you learned in the last lesson, bias-variance tradeoff. 

## Objectives

In this lab you will: 

- Demonstrate the tradeoff between bias and variance by way of fitting a machine learning model 

## Let's get started!

In this lab, you'll try to predict some movie revenues based on certain factors, such as ratings and movie year. Start by running the following cell which imports all the necessary functions and the dataset: 


```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import *
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_excel('movie_data_detailed_with_ols.xlsx')
df.head()
```


```python
# __SOLUTION__ 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import *
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_excel('movie_data_detailed_with_ols.xlsx')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>budget</th>
      <th>domgross</th>
      <th>title</th>
      <th>Response_Json</th>
      <th>Year</th>
      <th>imdbRating</th>
      <th>Metascore</th>
      <th>imdbVotes</th>
      <th>Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>13000000</td>
      <td>25682380</td>
      <td>21 &amp;amp; Over</td>
      <td>0</td>
      <td>2008</td>
      <td>6.8</td>
      <td>48</td>
      <td>206513</td>
      <td>4.912759e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>45658735</td>
      <td>13414714</td>
      <td>Dredd 3D</td>
      <td>0</td>
      <td>2012</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.267265e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>20000000</td>
      <td>53107035</td>
      <td>12 Years a Slave</td>
      <td>0</td>
      <td>2013</td>
      <td>8.1</td>
      <td>96</td>
      <td>537525</td>
      <td>1.626624e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>61000000</td>
      <td>75612460</td>
      <td>2 Guns</td>
      <td>0</td>
      <td>2013</td>
      <td>6.7</td>
      <td>55</td>
      <td>173726</td>
      <td>7.723381e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>40000000</td>
      <td>95020213</td>
      <td>42</td>
      <td>0</td>
      <td>2013</td>
      <td>7.5</td>
      <td>62</td>
      <td>74170</td>
      <td>4.151958e+07</td>
    </tr>
  </tbody>
</table>
</div>



Subset the `df` DataFrame to only keep the `'domgross'`, `'budget'`, `'imdbRating'`, `'Metascore'`, and `'imdbVotes'` columns. Use the `MinMaxScaler` to scale all these columns. 


```python
# Subset the DataFrame
df = None
```


```python
# __SOLUTION__ 
# Subset the DataFrame
df = df[['domgross', 'budget', 'imdbRating', 'Metascore', 'imdbVotes']]
```


```python
# Transform with MinMaxScaler
scale = None
transformed = None
pd_df = pd.DataFrame(transformed, columns=df.columns)
pd_df.head()
```


```python
# __SOLUTION__ 
# Transform with MinMaxScaler
scale = MinMaxScaler()
transformed = scale.fit_transform(df)
pd_df = pd.DataFrame(transformed, columns=df.columns)
pd_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>domgross</th>
      <th>budget</th>
      <th>imdbRating</th>
      <th>Metascore</th>
      <th>imdbVotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.055325</td>
      <td>0.034169</td>
      <td>0.839506</td>
      <td>0.500000</td>
      <td>0.384192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.023779</td>
      <td>0.182956</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.125847</td>
      <td>0.066059</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.183719</td>
      <td>0.252847</td>
      <td>0.827160</td>
      <td>0.572917</td>
      <td>0.323196</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.233625</td>
      <td>0.157175</td>
      <td>0.925926</td>
      <td>0.645833</td>
      <td>0.137984</td>
    </tr>
  </tbody>
</table>
</div>



## Split the data


- First, assign the predictors to `X` and the outcome variable, `'domgross'` to `y` 
- Split the data into training and test sets. Set the seed to 42 and the `test_size` to 0.25 


```python
# domgross is the outcome variable
X = None
y = None

X_train , X_test, y_train, y_test = None
```


```python
# __SOLUTION__ 
# domgross is the outcome variable
X = pd_df[['budget', 'imdbRating', 'Metascore', 'imdbVotes']]
y = pd_df['domgross']

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

## Fit a regression model to the training data and look at the coefficients


```python
# Your code 
linreg = None
```


```python
# __SOLUTION__ 
# Your code 
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg.coef_
```




    array([ 0.48419438, -0.2321452 ,  0.30774948,  0.18293653])



Calculate the mean squared error of the training set using this model: 


```python
# MSE of training set

```


```python
# __SOLUTION__ 
# MSE of training set
mean_squared_error(y_train, linreg.predict(X_train))
```




    0.026366234823542414



Calculate the mean squared error of the test set using this model: 


```python
# MSE of test set

```


```python
# __SOLUTION__ 
# MSE of test set
mean_squared_error(y_test, linreg.predict(X_test))
```




    0.06307564024771552



## Bias

Create a function `bias()` to calculate the bias of a model's predictions given the actual data: $Bias(\hat{f}(x)) = E[\hat{f}(x)-f(x)]$   
(The expected value can simply be taken as the mean or average value.)  


```python
import numpy as np
def bias(y, y_hat):
    pass
```


```python
# __SOLUTION__ 
import numpy as np
def bias(y, y_hat):
    return np.mean(y_hat - y)
```

## Variance
Create a function `variance()` to calculate the variance of a model's predictions: $Var(\hat{f}(x)) = E[\hat{f}(x)^2] - \big(E[\hat{f}(x)]\big)^2$


```python
def variance(y_hat):
    pass
```


```python
# __SOLUTION__ 
def variance(y_hat):
    return np.mean([yi**2 for yi in y_hat]) - np.mean(y_hat)**2
```

## Calculate bias and variance


```python
# Bias and variance for training set 
b = None
v = None
print('Bias: {} \nVariance: {}'.format(b, v))

# Bias: 2.901719268906659e-17 
# Variance: 0.027449331056376085
```


```python
# __SOLUTION__ 
# Bias and variance for training set 
b = bias(y_train, linreg.predict(X_train)) 
v = variance(linreg.predict(X_train)) 
print('Bias: {} \nVariance: {}'.format(b, v))
```

    Bias: 3.9110129276568017e-17 
    Variance: 0.02252739508558451



```python
# Bias and variance for test set 
b = None
v = None
print('Bias: {} \nVariance: {}'.format(b, v))

# Bias: 0.05760433770819166 
# Variance: 0.009213684542614783
```


```python
# __SOLUTION__ 
# Bias and variance for test set 
b = bias(y_test, linreg.predict(X_test)) 
v = variance(linreg.predict(X_test)) 
print('Bias: {} \nVariance: {}'.format(b, v))
```

    Bias: -0.02824089667424158 
    Variance: 0.0100422001582268


## Interpret the results


```python
# Your description here
```


```python
# __SOLUTION__ 
"""
These numbers indicate that the bias increases, but the variance
decreases. This indicates that the model is not overfitting, however
it might be overfitting.
"""
```




    '\nThese numbers indicate that the bias increases, but the variance\ndecreases. This indicates that the model is not overfitting, however\nit might be overfitting.\n'



## Overfit a new model 

Use `PolynomialFeatures` with degree 3 and transform `X_train` and `X_test`. 

**Important note:** By including this, you don't only take polynomials of single variables, but you also combine variables, eg:

$ \text{Budget} * \text{MetaScore} ^ 2 $

What you're essentially doing is taking interactions and creating polynomials at the same time! Have a look at how many columns we get using `np.shape()`! 



```python
# Your code here
poly = None

X_train_poly = None
X_test_poly = None
```


```python
# __SOLUTION__ 
poly = PolynomialFeatures(3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
```


```python
# Check the shape
```


```python
# __SOLUTION__ 
# Check the shape
np.shape(X_train_poly)
```




    (22, 35)



Fit a regression model to the training data: 


```python
# Your code here
linreg = None

```


```python
# __SOLUTION__ 
linreg = LinearRegression()
linreg.fit(X_train_poly, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



Calculate the mean squared error of the training set using this model:


```python
# MSE of training set 

```


```python
# __SOLUTION__ 
# MSE of training set
mean_squared_error(y_train, linreg.predict(X_train_poly))
```




    1.7542037443289215e-29



Calculate the mean squared error of the test set using this model:


```python
# MSE of test set
```


```python
# __SOLUTION__ 
# MSE of test set
mean_squared_error(y_test, linreg.predict(X_test_poly))
```




    0.49830553496326574



Calculate the bias and variance for the training set: 


```python
# Bias and variance for training set 
b = None 
v = None 
print('Bias: {} \nVariance: {}'.format(b, v))
# Bias: -2.5421584029769207e-16 
# Variance: 0.07230707736656222
```


```python
# __SOLUTION__ 
# Bias and variance for training set 
b = bias(y_train, linreg.predict(X_train_poly))
v = variance(linreg.predict(X_train_poly))
print('Bias: {} \nVariance: {}'.format(b, v))
```

    Bias: -3.8952427142388303e-16 
    Variance: 0.0488936299091263


Calculate the bias and variance for the test set: 


```python
# Bias and variance for test set 
b = None 
v = None 
print('Bias: {} \nVariance: {}'.format(b, v))
```


```python
# __SOLUTION__ 
# Bias and variance for test set 
b = bias(y_test, linreg.predict(X_test_poly)) 
v = variance(linreg.predict(X_test_poly)) 
print('Bias: {} \nVariance: {}'.format(b, v))
```

    Bias: -0.17528690868564342 
    Variance: 0.3172819263811148


## Interpret the overfit model


```python
# Your description here
```


```python
# __SOLUTION__
# The bias and variance for the test set both increased drastically in the overfit model.
```

## Level Up (Optional)

In this lab we went from 4 predictors to 35 by adding polynomials and interactions, using `PolynomialFeatures`. That being said, where 35 leads to overfitting, there are probably ways to improve by adding just a few polynomials. Feel free to experiment and see how bias and variance improve!

## Summary

This lab gave you insight into how bias and variance change for a training and a test set by using both simple and complex models. 
