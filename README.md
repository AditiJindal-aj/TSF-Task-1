# TSF-Task-1
TASK 1: PREDICTION USING SUPERVISED MACHINE LEARNING
GRIP MAY 2021
In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.
Author: Debasmita Majumder
IMPORTING THE STANDARD LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


%matplotlib inline
url="http://bit.ly/w-data"
data=pd.read_csv(url)
print('Data successfully loaded')
Data successfully loaded
data.head(10)
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
data.describe()
Hours	Scores
count	25.000000	25.000000
mean	5.012000	51.480000
std	2.525094	25.286887
min	1.100000	17.000000
25%	2.700000	30.000000
50%	4.800000	47.000000
75%	7.400000	75.000000
max	9.200000	95.000000
data.shape
(25, 2)
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 528.0 bytes
Plotting the data
font1 = {'family':'Calibri','color':'blue','size':20}
font2 = {'family':'serif','color':'green','size':15}
data.plot(x='Hours',y='Scores',style='o',c='red')
plt.title('Hours vs Score',fontdict=font1)
plt.xlabel('Hours studied',fontdict=font2)
plt.ylabel('Score obtained',fontdict=font2)
plt.show()

data.corr()
Hours	Scores
Hours	1.000000	0.976191
Scores	0.976191	1.000000
From above plot, it is concluded that there is a linear relationship between the Hours and scores.
Cleaning the data
data.isnull().sum()
Hours     0
Scores    0
dtype: int64
Training and testing the model
x=(data['Hours'].values).reshape(-1,1)
y=data['Scores'].values
x
array([[2.5],
       [5.1],
       [3.2],
       [8.5],
       [3.5],
       [1.5],
       [9.2],
       [5.5],
       [8.3],
       [2.7],
       [7.7],
       [5.9],
       [4.5],
       [3.3],
       [1.1],
       [8.9],
       [2.5],
       [1.9],
       [6.1],
       [7.4],
       [2.7],
       [4.8],
       [3.8],
       [6.9],
       [7.8]])
y
array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30,
       24, 67, 69, 30, 54, 35, 76, 86], dtype=int64)
from sklearn.model_selection import train_test_split  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print('splitting is done')
splitting is done
from sklearn.linear_model import LinearRegression
regn = LinearRegression()
regn.fit(x_train,y_train)
print('Training is done')
Training is done
print('Intercept value is:',regn.intercept_)
print('Linear coefficient is:',regn.coef_)
Intercept value is: 2.018160041434683
Linear coefficient is: [9.91065648]
# Plotting the regression line
line = regn.coef_*x+regn.intercept_

# Plotting for the test data
plt.scatter(x, y,c='red')
plt.title('Linear Regression vs trained model',fontdict=font1)
plt.xlabel('Hours studied',fontdict=font2)
plt.ylabel('Score obtained',fontdict=font2)
plt.plot(x, line);
plt.show()

#to predict scores of testing data
y_pred = regn.predict(x_test)
y_pred
array([16.88414476, 33.73226078, 75.357018  , 26.79480124, 60.49103328])
Comparing actual vs predicted values
df= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
plt.scatter(y_test,y_pred,c='green')
plt.show()

Solution for given problem statement
What will be predicted score if a student studies for 9.25 hrs/ day?

hours=9.25
pred_score=regn.predict([[hours]])
print("Number of hours = {}".format(hours))
print("Predicted Score = {}".format(pred_score[0]))
Number of hours = 9.25
Predicted Score = 93.69173248737538
Evaluating the Model
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))
Mean Absolute Error: 4.183859899002975
 
