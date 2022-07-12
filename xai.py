# code for ALE
import matplotlib as matplotlib

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from alibi.explainers import ALE, plot_ale
import pickle

#p = r".\files_for_training_model\model.pkl"
#lr = pickle.load(open(p, 'rb'))
# X_train = pickle.load(open(filename, 'rb'))

import mlflow


def ale_for_model():

    logged_model = 'wasbs://artifacts@wnsmlopstrgci.blob.core.windows.net/17/e11ebe5dacae409caebd369b383c9c35/artifacts/model'

    # Load model as a PyFuncModel.
    lr = mlflow.pyfunc.load_model(logged_model)

    filename = r".\files_for_training_model\X_train.pickle"
    X_train = pickle.load(open(filename, 'rb'))


    feature_names = X_train.columns.to_list()
    lr_ale = ALE(lr.predict, feature_names=feature_names, target_names=["heart.disease"])

    print("## Feature Effects: Motivation")

    print(type(feature_names))

    X_train = X_train.to_numpy()
    lr_exp = lr_ale.explain(X_train)
    plot_ale(lr_exp, fig_kw={'figwidth':6, 'figheight': 3});

    print("done")

