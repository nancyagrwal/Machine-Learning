# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 02:18:12 2017

@author: nancy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io as sio
import seaborn as sb


def calculate_error(w,X,y):
    na = classify_error(w,X)
    me = np.mean(na==y)
    return 1 - me

def classify_error(w, X):
    return np.round(invertedCode(w, X))


def inverse_log_odds(z):
    return  1.0/(1.0 + np.exp(-z))

def invertedCode(w, X):
    return inverse_log_odds(X.dot(w[:, None]))


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



def data_train(X,y):
    leng, d = X.shape
    p = 1
    weight = np.zeros(d)
    p2 = calculate_cost
    # using optimize function:
    w_optimal = optimize.fmin_bfgs(p2,weight,args=(X, y, p), fprime = calculate_gradient)
    return w_optimal



def plot_prediction(weight,X,y,title = 'train set'):
    data = pd.DataFrame({"x1": X[:,0].tolist(), "x2": X[:,1].tolist(), "y": y[:,0].tolist()})
    sb.lmplot("x1", "x2", hue="y", data=data, fit_reg=False, ci=False)
    mini = np.min(X[:,1])
    maxi = np.max(X[:,1])
    wt_approx = -weight[1] / weight[0]
    min_x = mini * wt_approx
    max_y = maxi * wt_approx
    plt.plot([min_x, max_y], [mini, maxi])
    plt.title(title)
    plt.show()


data = sio.loadmat("HW2_Data/data2.mat")
X_train = data['X_trn']
y_train = data['Y_trn']
X_test = data['X_tst']
y_test = data['Y_tst']
wt_vect = data_train(X_train, y_train)

print("Obtained Weight Vec: ", wt_vect)

print("Calculated Training Err: ", calculate_error(wt_vect, X_train, y_train))
plot_prediction(wt_vect, X_train, y_train,"Training set")

print("Calculated Test Err: ", calculate_error(wt_vect, X_test, y_test))
plot_prediction(wt_vect, X_test, y_test, "test set")