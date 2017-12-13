import matplotlib.pyplot as plt
import numpy as np
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

X,Y = inputProcessing(Y_train)
Xtst,Ytst = inputProcessing(Y_test)

call = LogisticRegression()
call.fit(X, Y)
print("mean accuracy on the given test data and labels is..." , call.score(Xtst, Ytst))
