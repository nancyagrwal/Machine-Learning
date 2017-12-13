"""
@author: nancy
"""
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
dataset1 = sio.loadmat('HW1_Data/dataset1')
import q10_1 as reg

X_train, y_train = dataset1['X_trn'], dataset1['Y_trn']
X_train.shape, y_train.shape
X_test, y_test = dataset1['X_tst'], dataset1['Y_tst']
X_test.shape, y_test.shape


# add intercept term:
X_train, X_test = np.hstack((np.ones((X_train.shape[0], 1)), X_train)), np.hstack((np.ones((X_test.shape[0], 1)), X_test))
theta = reg.simpleLinearRegression(X_train, y_train)
y_train_predict = np.dot(X_train, theta)

#plotPrediction(y_train, y_train_predict, 'Training set')
y_test_predict = np.dot(X_test, theta)
#plotPrediction(y_test, y_test_predict, 'Test set')

num_features, mses_train, mses_test = [], [], []

#for bias n belongs {2,3,5} :
X_train2, X_test2 = np.hstack((X_train, X_train[:, 1:] **2)), np.hstack((X_test, X_test[:, 1:] **2))
theta2 = reg.simpleLinearRegression(X_train2, y_train)
y_train_predict2 = np.dot(X_train2, theta2)
y_test_predict2 = np.dot(X_test2, theta2)


X_train3, X_test3 = np.hstack((X_train, X_train[:, 1:] **2, X_train[:, 1:] **3)), np.hstack((X_test, X_test[:, 1:] **2, X_test[:, 1:] **3))
theta3 = reg.simpleLinearRegression(X_train3, y_train)
y_train_predict3 = np.dot(X_train3, theta3)
y_test_predict3 = np.dot(X_test3, theta3)


X_train5 = np.hstack((X_train, X_train[:, 1:] **2, X_train[:, 1:] **3, X_train[:, 1:] **4, X_train[:, 1:] **5))
X_test5 = np.hstack((X_test, X_test[:, 1:] **2, X_test[:, 1:] **3, X_test[:, 1:] **4, X_test[:, 1:] **5))
theta5 = reg.simpleLinearRegression(X_train5, y_train)
y_train_predict5 = np.dot(X_train5, theta5)
y_test_predict5 = np.dot(X_test5, theta5)


print ('RESULTS for linear regression using Closed Form')

print ("____________ optimal Theta ____________")
print ("n = 1")
num_features.append(X_train.shape[1])
print ("theta = {}".format(np.transpose(theta)))

print ("n = 2")
num_features.append(X_train2.shape[1])
print ("theta = {}".format(np.transpose(theta2)))

print ("n = 3")
num_features.append(X_train3.shape[1])
print ("theta = {}".format(np.transpose(theta3)))

print ("n = 5")
num_features.append(X_train5.shape[1])
print ("thata = {}".format(np.transpose(theta5)))

# ############################### ERRORS data ##################################:

print ("__________________ errors using Closed Form:_________________") 
print ("n = 1, training:" )
error_1_trn =  reg.MSE(y_train, y_train_predict)
mses_train.append(error_1_trn)
print (error_1_trn)

print ("n = 1, test:" )
error_1_tst = reg.MSE(y_test, y_test_predict)
mses_test.append(error_1_tst)
print (error_1_tst)
    
print ("============================")

print ("n = 2, training:" )
error_2_trn =  reg.MSE(y_train, y_train_predict2)
mses_train.append(error_2_trn)
print (error_2_trn)

print ("n = 2, test:" )
error_2_tst = reg.MSE(y_test, y_test_predict2)
mses_test.append(error_2_tst)
print (error_2_tst)

print ("============================")

print ("n = 3, training:" )
error_3_trn =  reg.MSE(y_train, y_train_predict3)
mses_train.append(error_3_trn)
print (error_3_trn)

print ("n = 3, test:" )
error_3_tst = reg.MSE(y_test, y_test_predict3)
mses_test.append(error_3_tst)
print (error_3_tst)

print ("============================")

print ("n = 5, training:")
error_5_trn = reg.MSE(y_train, y_train_predict5)
mses_train.append(error_5_trn)
print (error_5_trn)

print ("n = 5, test:")
error_5_tst = reg.MSE(y_train, y_train_predict5)
mses_test.append(error_5_tst)
print (error_5_tst)

print ("============================")

plt.plot(num_features, mses_train, marker = '.')
plt.plot(num_features, mses_test, marker = '.')
plt.legend(['train', 'test'])
plt.xlabel('number of features')
plt.ylabel('MSE')    	
plt.title('feature vs MSE')
    
# initialize data:
alpha = 0.01
iters = 1000
minibatch_size = 5
theta0 = np.matrix(np.zeros(X_train.shape[1]))
theta_test = np.matrix(np.zeros(X_test.shape[1]))

print ('RESULTS for linear regression using Stochastic Gradient Descent')

print ("____________ optimal Theta for training set ____________")
print("size of batch = 5:")
thetas0, cost_train0 = reg.stochasticGradientDescent(X_train, y_train, theta0, alpha, iters, 5)
print ("theta = {}".format(np.transpose(thetas0)))

print ("____________ optimal Theta for test set ____________")
print("size of batch = 5:")
thetas1, cost_test1 = reg.stochasticGradientDescent(X_test, y_test, theta_test, alpha, iters, 5)
print ("theta = {}".format(np.transpose(thetas1)))

print("size of batch = 23:")
thetas2, cost_test2 = reg.stochasticGradientDescent(X_test, y_test, theta_test, alpha, iters, 23)
print ("theta = {}".format(np.transpose(thetas2)))


print("size of batch = 50:")
thetas3, cost_test3 = reg.stochasticGradientDescent(X_test, y_test, theta_test, alpha, iters, 50)
print ("theta = {}".format(np.transpose(thetas3)))


print("---------------------Errors----------------" )
print("----------------------Training data---------")
training_error = reg.computeCost(X_train, y_train, thetas0)
print('Training error is...' , training_error)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost_train0, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. training Epoch')

print("----------------------Testing data---------")
print("size of batch = 5:")
testing_error = reg.computeCost(X_test, y_test, thetas1)
print('Testing error is...' , testing_error)


print("size of batch = 23:")
testing_error2 = reg.computeCost(X_test, y_test, thetas2)
print('Testing error is...' , testing_error2)


print("size of batch = 50:")
testing_error3 = reg.computeCost(X_test, y_test, thetas3)
print('Testing error is...' , testing_error3)



fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost_test3, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Test Epoch')











    