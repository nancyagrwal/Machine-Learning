from numpy import array
from scipy.io import loadmat
from sklearn import svm

dataDict = dict()
dataDict = loadmat("HW2_Data/data1")
X_train = dataDict['X_trn']
X_test = dataDict['X_tst']
Y_train = dataDict['Y_trn']
Y_test = dataDict['Y_tst']

X_train = X_train.astype(float, copy=False)
X_test = X_test.astype(float, copy=False)
Y_train = Y_train.astype(int, copy=False)
Y_test = Y_test.astype(int, copy=False)

givenKernel = ['linear', 'poly', 'rbf', 'sigmoid']

for i in givenKernel:
    c = svm.SVC(kernel=i)
    k = array(Y_train.transpose().tolist()[0])
    data_fitting = c.fit(X_train,k)
    resuts = c.predict(X_test)
    print(data_fitting)
    print(data_fitting._get_coef())
    print(resuts)
    print()


          
        