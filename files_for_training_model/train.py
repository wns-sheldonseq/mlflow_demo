# https://youtu.be/bluclMxiUkA
"""
Multiple Linear Regression uses several explanatory variables to predict the outcome of a response variable.
There are a lot of variables and multiple linear regression is designed to create a model 
based on all these variables. 

#Dataset link:
https://cdn.scribbr.com/wp-content/uploads//2020/02/heart.data_.zip?_ga=2.217642335.893016210.1598387608-409916526.1598387608

#Heart disease
The effect that the independent variables biking and smoking 
have on the dependent variable heart disease 

the percentage of people biking to work each day, the percentage of people smoking, 
and the percentage of people with heart disease in an imaginary sample of 500 towns.


"""
import os
import warnings
import sys

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import logging

#mlflow.set_tracking_uri("http://10.31.8.38:5000")
mlflow.set_experiment("/Users/u392578@wns.com/Exp_01")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


df = pd.read_csv('heart_data.csv')
#print(df.head())

df = df.drop("Unnamed: 0", axis=1)
#A few plots in Seaborn to understand the data

#sns.lmplot(x='biking', y='heart.disease', data=df)  
#sns.lmplot(x='smoking', y='heart.disease', data=df)  


x_df = df.drop('heart.disease', axis=1)
y_df = df['heart.disease']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

#pickle X_ train for ALE
#save lode X_train
filename = 'X_train.pickle'
pickle.dump(X_train, open(filename, 'wb'))



from sklearn import linear_model

with mlflow.start_run(experiment_id="4281740375320786"):
#Create Linear Regression object
    model = linear_model.LinearRegression()

#Now let us call fit method to train the model using independent variables.
#And the value that needs to be predicted (Images_Analyzed)

    model.fit(X_train, y_train) #Indep variables, dep. variable to be predicted
    print(model.score(X_train, y_train))  #Prints the R^2 value, a measure of how well


    prediction_test = model.predict(X_test)    
    print(y_test, prediction_test)
    print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)

    pickle.dump(model, open('model.pkl','wb'))

    model = pickle.load(open('model.pkl','rb'))
    #print(model.predict([[20.1, 56.3]]))

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
    if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(model, "model", registered_model_name="Linear-Regression")
    else:
        mlflow.sklearn.log_model(model, "model")
#Model is ready. Let us check the coefficients, stored as reg.coef_.
#These are a, b, and c from our equation. 
#Intercept is stored as reg.intercept_
#print(model.coef_, model.intercept_)

#All set to predict the number of images someone would analyze at a given time
#print(model.predict([[13, 2, 23]]))
