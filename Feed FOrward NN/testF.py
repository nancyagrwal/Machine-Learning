from pca import PCA
import numpy as np
from matplotlib import pyplot as plt
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

# apply PCA to reduce dimenions = 2
X = PCA(X.T, 2)[-1].T

# Plotting the graph for data visualization:
plt.style.use("classic")
colors = ['b', 'lime', 'c', 'r', 'y', 'm', 'k', 'teal', 'silver', 'aqua']
figure, axis = plt.subplots()
for i in range(10):
    nodes = np.array([X[j] for j in range(len(X)) if Y[j] == i])
    axis.scatter(nodes[:, 0], nodes[:, 1], s=16, c=colors[i])
plt.show()

