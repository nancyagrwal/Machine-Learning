from pca import PCA
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import feedForwardNeuralNetwork
from sklearn.linear_model import LogisticRegression
from skimage.transform import resize
import scipy.io as scio

mat = scio.loadmat('./ExtYaleB10.mat')
Y_test = mat['test']
Y_train = mat['train']

def imageResizing(data):
    resizedDatasET = []
    for img in data:
        resizedDatasET.append(resize(img, (20, 17), mode='constant'))
    resizedDatasET = np.array(resizedDatasET)
    return resizedDatasET

def imageReshaping(data):
    dimension = data.shape[1] * data.shape[2]
    return data.reshape(data.shape[0], dimension)

def inputProcessing(data):
    X = [];Y = []
    for i in range(len(data[0])):
        people_count = data[0][i].T
        for j in range(len(people_count)):
            X.append(people_count[j].T);Y.append(i)
    X = np.array(X);Y = np.array(Y)
    fig, axis = plt.subplots(figsize=(12,8))
    axis.imshow(X[1], cmap='gray')
    X = imageResizing(X)
    X = imageReshaping(X)
    return X, Y

def findError(y, ypred):
    return (ypred != y).sum().astype(float) / len(ypred)

X,Y = inputProcessing(Y_train)

# first apply PCA to reduce dimensions to 100
X = PCA(X.T, 100)[-1].T
print("Dimenions of training data = ", X.shape)
print("Dimenions of test data = " , Y.shape)

# Feed forward neural net:
for n in [10, 30 , 100]:
    S = [100, n, 10]
    trainedmodel = feedForwardNeuralNetwork.FeedForwardNeuralNetwork(S)
    activFuncftions = ['tanh' , 'sigmoid' ,'relu']
    for av in activFuncftions:
        W,b = trainedmodel.train(X, Y, av , max_passes=10000, epsilon=0.01, reg_lambda=0.01, loss_printYN=True)
        #print("Weights are...." , W)
        #print("Biases are.....", b)
        Xtst,Ytst = inputProcessing(Y_test)
        Xtst = PCA(Xtst.T, 100)[-1].T
        print("Dimenions of test data = " , Xtst.shape)
        print("Dimenions of test data = " , Ytst.shape)
        ypred = []
        for x in Xtst:
                pred = trainedmodel.predict(x,"tanh")
                ypred.append(pred[0])
              
        print("Classification Error: %.2f%", av , n, findError(Ytst, ypred))
        print("****************************************************")


# SVM
classifierFunc2 = LinearSVC()
classifierFunc2.fit(X, Y)
print("SVM accuracy: ", classifierFunc2.score(Xtst, Ytst))

# Logistic Regression
classifierFunc3 = LogisticRegression()
classifierFunc3.fit(X, Y)
print("LR accuracy: ", classifierFunc3.score(Xtst, Ytst))
