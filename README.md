# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1: Start

Step-2: Import the necessary python packages

Step-3: Read the dataset.

Step-4: Define X and Y array.

Step-5: Define a function for costFunction,cost and gradient.

Step-6: Define a function to plot the decision boundary and predict the Regression value.

Step-7: End

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: \ESHWAR T
RegisterNumber: 212223230054
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
#dropping the serial no and salary col
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

#categorising col for further labeling
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset

#selecting the features and labels
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
#initialize the model parameters.
theta=np.random.randn(X.shape[1])
y=Y
#define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#Define the Loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return  -np.sum(y*np.log(h)+(1-y)*log(1-h))

#Define the gradient descent algorithm
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
#train the model
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
#Make predictions
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>0.5,1,0)
    return y_pred
y_pred=predict(theta,X)

#evaluate the model
accuracy=np.mean(y_pred.flatten()==y)
print('Accuracy:',accuracy)

print(y_pred)
print("y_pred : \n ",Y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(" y_prednew : \n ",y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(" y_prednew : ",y_prednew)

```

## Output:

![Screenshot 2024-10-16 160651](https://github.com/user-attachments/assets/16b2044f-7c17-410d-bd10-3377b41fa42b)

![Screenshot 2024-10-16 160646](https://github.com/user-attachments/assets/705159bf-2354-4140-805d-ece5555d8da8)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

