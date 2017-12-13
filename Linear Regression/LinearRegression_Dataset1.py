"""
@author: nancy
"""
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

dataset1 = sio.loadmat('./dataset1')
X_train, y_train = dataset1['X_trn'], dataset1['Y_trn']
X_train.shape, y_train.shape
X_test, y_test = dataset1['X_tst'], dataset1['Y_tst']
X_test.shape, y_test.shape

X_train1, X_test1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train)), np.hstack((np.ones((X_test.shape[0], 1)), X_test))
X_train2, X_test2 = np.hstack((X_train, X_train[:, 1:] **2)), np.hstack((X_test, X_test[:, 1:] **2))
X_train3, X_test3 = np.hstack((X_train, X_train[:, 1:] **2, X_train[:, 1:] **3)), np.hstack((X_test, X_test[:, 1:] **2, X_test[:, 1:] **3))
X_train5 = np.hstack((X_train, X_train[:, 1:] **2, X_train[:, 1:] **3, X_train[:, 1:] **4, X_train[:, 1:] **5))
X_test5 = np.hstack((X_test, X_test[:, 1:] **2, X_test[:, 1:] **3, X_test[:, 1:] **4, X_test[:, 1:] **5))


def simpleLinearRegression(X, y):
    # Theta = (X^T*X)^{-1}*X^T*y
    var = np.dot(X.T, X)
    cov = np.dot(X.T, y)
    theta = np.dot(np.linalg.inv(var), cov)
    return theta

def RidgeRegression(X, y, la=0.1):
    d = X.shape[1]
    return np.linalg.solve(X.T.dot(X) + la * np.eye(d), X.T.dot(y))
  
    
def MSE(y, predictedY):
    return np.mean((y - predictedY) ** 2)

def plotPrediction(y, predictedY, title = 'train set'):
    plt.scatter(y, predictedY)
    plt.plot([min(y), max(y)], [min(y), max(y)], '--')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.title(title)
    
    
# add intercept term:
X_train, X_test = np.hstack((np.ones((X_train.shape[0], 1)), X_train)), np.hstack((np.ones((X_test.shape[0], 1)), X_test))
theta = simpleLinearRegression(X_train, y_train)
y_train_predict = np.dot(X_train, theta)
y_test_predict = np.dot(X_test, theta)


#for bias n belongs {2,3,5} :
X_train2, X_test2 = np.hstack((X_train, X_train[:, 1:] **2)), np.hstack((X_test, X_test[:, 1:] **2))
theta2 = simpleLinearRegression(X_train2, y_train)
y_train_predict2 = np.dot(X_train2, theta2)
y_test_predict2 = np.dot(X_test2, theta2)

# ############################### ERRORS data ##################################:

error_1_trn =  MSE(y_train, y_train_predict)
error_1_tst = MSE(y_test, y_test_predict)
error_2_trn =  MSE(y_train, y_train_predict2)
error_2_tst = MSE(y_test, y_test_predict2)


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# initialize data:
alpha = 0.01
iters = 1000
minibatch_size = 5
theta0 = np.matrix(np.zeros(X_train.shape[1]))
theta_test = np.matrix(np.zeros(X_test.shape[1]))

def stochasticGradientDescent(X, y, theta, alpha, iters, minibatch_size):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.ravel().shape[1]
    cost = np.zeros(iters)
    
    for i in range(iters):
        
        for picked_batch in range(0, X.shape[0], minibatch_size):
            # Get pair of (X, y) of the current minibatch/chunk
            X_mini = X[picked_batch:picked_batch + minibatch_size]
            y_mini = y[picked_batch:picked_batch + minibatch_size]
            error = (X_mini * theta.T) - y_mini
            for j in range(parameters):
                term = np.multiply(error, X_mini[:,j])
                temp[0,j] = theta[0,j] - ((alpha / len(X_mini)) * np.sum(term))

            theta = temp
            
        cost[i] = computeCost(X, y, theta)
        
        if i % 50 == 0:
            print("Loss iteration",i,": ",cost[i])
        
    return theta, cost


thetas0, cost_train0 = stochasticGradientDescent(X_train, y_train, theta0, alpha, iters, 5)
thetas1, cost_test1 = stochasticGradientDescent(X_test, y_test, theta_test, alpha, iters, 5)
training_error = computeCost(X_train, y_train, thetas0)
testing_error = computeCost(X_test, y_test, thetas1)








    