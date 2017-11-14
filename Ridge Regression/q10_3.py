"""
@author: nancy
"""

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.io as sio
from scipy import array, inner, average, linspace, array_split, eye, dot
from scipy.linalg import inv
dataset1 = sio.loadmat('HW1_Data/dataset2')
import numpy as np
import matplotlib.pyplot as plt


X_train = dataset1['X_trn']
Y_train = dataset1['Y_trn']
X_test = dataset1['X_tst']
Y_test = dataset1['Y_tst']
kFoldSet = [2, 10, len(X_train)]
basisFunctionGiven = [2, 3, 5]
minRange = 0.001
maxRange = 0.1
intervals = 100
lamdaList = linspace(minRange, maxRange, intervals)
MAX_VALUE = float('inf')

def RidgeRegression(X, y, la=0.1):
    d = X.shape[1]
    return np.linalg.solve(X.T.dot(X) + la * np.eye(d), X.T.dot(y))

def MSE(y, predictedY):
    return np.mean((y - predictedY) ** 2)

# add intercept term:
X_train, X_test = np.hstack((np.ones((X_train.shape[0], 1)), X_train)), np.hstack((np.ones((X_test.shape[0], 1)), X_test))
theta = RidgeRegression(X_train, Y_train)
Y_train_predict = np.dot(X_train, theta)
#plotPrediction(y_train, y_train_predict, 'Training set')
Y_test_predict = np.dot(X_test, theta)
#plotPrediction(y_test, y_test_predict, 'Test set')

X_train2, X_test2 = np.hstack((X_train, X_train[:, 1:] **2)), np.hstack((X_test, X_test[:, 1:] **2))
theta2 = RidgeRegression(X_train2, Y_train)
Y_train_predict2 = np.dot(X_train2, theta2)
Y_test_predict2 = np.dot(X_test2, theta2)


X_train3, X_test3 = np.hstack((X_train, X_train[:, 1:] **2, X_train[:, 1:] **3)), np.hstack((X_test, X_test[:, 1:] **2, X_test[:, 1:] **3))
theta3 = RidgeRegression(X_train3, Y_train)
Y_train_predict3 = np.dot(X_train3, theta3)
Y_test_predict3 = np.dot(X_test3, theta3)


X_train5 = np.hstack((X_train, X_train[:, 1:] **2, X_train[:, 1:] **3, X_train[:, 1:] **4, X_train[:, 1:] **5))
X_test5 = np.hstack((X_test, X_test[:, 1:] **2, X_test[:, 1:] **3, X_test[:, 1:] **4, X_test[:, 1:] **5))
theta5 = RidgeRegression(X_train5, Y_train)
Y_train_predict5 = np.dot(X_train5, theta5)
Y_test_predict5 = np.dot(X_test5, theta5)

num_features, mses_train, mses_test = [], [], []

print ('RESULTS for ridge regression using Closed Form')

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
error_1_trn =  MSE(Y_train, Y_train_predict)
mses_train.append(error_1_trn)
print (error_1_trn)

print ("n = 1, test:" )
error_1_tst = MSE(Y_test, Y_test_predict)
mses_test.append(error_1_tst)
print (error_1_tst)
    
print ("============================")

print ("n = 2, training:" )
error_2_trn =  MSE(Y_train, Y_train_predict2)
mses_train.append(error_2_trn)
print (error_2_trn)

print ("n = 2, test:" )
error_2_tst = MSE(Y_test, Y_test_predict2)
mses_test.append(error_2_tst)
print (error_2_tst)

print ("============================")

print ("n = 3, training:" )
error_3_trn =  MSE(Y_train, Y_train_predict3)
mses_train.append(error_3_trn)
print (error_3_trn)

print ("n = 3, test:" )
error_3_tst = MSE(Y_test, Y_test_predict3)
mses_test.append(error_3_tst)
print (error_3_tst)

print ("============================")

print ("n = 5, training:")
error_5_trn = MSE(Y_train, Y_train_predict5)
mses_train.append(error_5_trn)
print (error_5_trn)

print ("n = 5, test:")
error_5_tst = MSE(Y_train, Y_train_predict5)
mses_test.append(error_5_tst)
print (error_5_tst)

print ("============================")


def computeCost(X, y, theta,lamda):
    inner = np.power(((X * theta.T) - y), 2)
    return (np.sum(inner) + (lamda * (theta * theta.T)[0,0])) / (2 * len(X))


# initialize data:
alpha = 0.01
iters = 1000
minibatch_size = 5
theta0 = np.matrix(np.zeros(X_train.shape[1]))
theta_test = np.matrix(np.zeros(X_test.shape[1]))

def stochasticGradientDescent(X, y, theta, alpha, iters, minibatch_size,lamda):
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
                temp[0,j] = theta[0,j] - ((alpha / len(X_mini)) * (np.sum(term) + (lamda*theta[0,j])))

            theta = temp
            
        cost[i] = computeCost(X, y, theta,lamda)
        
        if i % 50 == 0:
            print("Loss iteration",i,": ",cost[i])
        
    return theta, cost

print ('RESULTS for ridge regression using Stochastic Gradient Descent')

print ("____________ optimal Theta for training set ____________")
print("size of batch = 5:")
thetas0, cost_train0 = stochasticGradientDescent(X_train, Y_train, theta0, alpha, iters, 5,0.01)
print ("theta = {}".format(np.transpose(thetas0)))

print ("____________ optimal Theta for test set ____________")
print("size of batch = 5:")
thetas1, cost_test1 = stochasticGradientDescent(X_test, Y_test, theta_test, alpha, iters, 5,0.01)
print ("theta = {}".format(np.transpose(thetas1)))

print("---------------------Errors----------------" )
print("----------------------Training data---------")
training_error = computeCost(X_train, Y_train, thetas0,0.1)
print('Training error is...' , training_error)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost_train0, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. training Epoch')

print("----------------------Testing data---------")
print("size of batch = 5:")
testing_error = computeCost(X_test, Y_test, thetas1,0.1)
print('Testing error is...' , testing_error)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost_test1, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Test Epoch')



def returnL2Norm(data, lbl, weightVector):
    Ypredict = dot(data, weightVector)
    deltaY = lbl - Ypredict
    l2 = inner(deltaY.transpose(), deltaY.transpose())[0][0]
    return l2


def findModelWeightVector(lamda , X_train, Y_train):
    weightVector = dot(X_train.transpose(), X_train)
    weightVector = weightVector + lamda * eye(weightVector.shape[0])
    weightVector = dot(inv(weightVector), X_train.transpose())
    weightVector = dot(weightVector, Y_train)
    return weightVector

# define the basis function:
def phi(r,n):
        res = list()
        res.append(1)
        for bias in range(1, n + 1):
            for element in r:
                res.append(element ** bias)
                return array(res)


# training data with basis function
def basisFunctiontraining(train_data, dof):
    newData = list()
    for bias in train_data:
            newData.append(phi(bias, dof))
    newData = array(newData)        
    return newData                  


def predict(weightVector, dof):
    TestDatanew = basisFunctiontraining(X_test, dof)
    test_err = returnL2Norm(TestDatanew, Y_test, weightVector)
    print("Reported Testing error = " + str(test_err))
    
    

def validateKFold(choseholdOut, folds, block_data, trnLabel):
    hoData = block_data[choseholdOut]
    hoLabel = trnLabel[choseholdOut]
    newLabel = list()
      
    newData = list()
  
    for i in range(folds):
        if i == choseholdOut:
            continue
        newData.extend(block_data[i])
        newLabel.extend(trnLabel[i])
    newData = array(newData)
    newLabel = array(newLabel)
    return (newData, newLabel, hoData, hoLabel)




def KfoldTraining(dof):
    for choseholdOut in kFoldSet:
        weightVector = performKFoldValidation(choseholdOut, dof)
        print("Chosing fold:Training with " + str(choseholdOut) +" folds")
        print("Optimal weight vector =")
        print(weightVector.transpose())
        predict(weightVector, dof)
        print()



# using k-fold validation:
def performKFoldValidation(numberOfFolds, dof):
   
    feasLamda = None
    trainingData = X_train
    newTrainingdata = basisFunctiontraining(trainingData, dof)
    minError = MAX_VALUE
    trnLabel = array_split(Y_train, numberOfFolds)
    blocksX_train = array_split(newTrainingdata, numberOfFolds)
    modifiedWVector = None
     
    for l in lamdaList:
        errorPerLmda = list()
        for choseholdOut in range(numberOfFolds):
                KFolddata = validateKFold(choseholdOut,numberOfFolds,blocksX_train,trnLabel)
                
                newData, newLabel, hdData, hoLabel = KFolddata
                
                weightVector = findModelWeightVector(l,newData,newLabel)
                
                finalErr = returnL2Norm(hdData, hoLabel, weightVector)
                
                errorPerLmda.append(finalErr)
                
        errorPerLmda = average(errorPerLmda)
        # error:
        if minError > errorPerLmda:
            minError = errorPerLmda
            
            modifiedWVector = weightVector
            
            feasLamda = l
    print("optimal Value for lambda: " + str(feasLamda))
    print("Obtained Training Error = " + str(minError))
    return modifiedWVector


for dof in basisFunctionGiven:
    print("Degree of Basis Function = " + str(dof))
    print()
    KfoldTraining(dof)
    print()