# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 04:13:12 2017

@author: nancy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io as sio
import seaborn as sb

#Write down a code in Python whose input is a training dataset {(x1, y1), . . . ,(xN , yN )} and its
#output is the weight vector w in the logistic regression model y = σ(wTx).

def calculate_error(w,X,y):
    na = classify_error(w,X)
    me = np.mean(na==y)
    return 1 - me

def classify_error(w, X):
    return np.round(invertedCode(w, X))

def invertedCode(w, X):
    return inverse_log_odds(X.dot(w[:, None]))

def inverse_log_odds(z):
    return  1.0/(1.0 + np.exp(-z))


# the below functions would be used for training the data
def data_train(X,y):
    leng, d = X.shape
    p = 1
    weight = np.zeros(d)
    p2 = calculate_cost
    # using optimize function:
    w_optimal = optimize.fmin_bfgs(p2,weight,args=(X, y, p), fprime = calculate_gradient)
    return w_optimal

def calculate_cost(weight, X, y,p):
    h = invertedCode(weight,X)
    r = 1e-23
    length = float(len(y))
    h[h < r] = r
    h[(1 - r < h) & (h < 1 + r)] = 1 - r
    C = (float(p) / 2) * weight**2
    c1 = y * np.log(h)
    cost =  c1 + (1 - y) * np.log(1 - h)
    return ((-sum(cost) + sum(C)) / length)[0]


def calculate_gradient(weight,X,y,p):
    length = len(y)
    sigm = invertedCode(weight,X)
    norm = float(p) * weight/length
    return np.ndarray.flatten((sigm-y).T.dot(X) / length + norm.T)



