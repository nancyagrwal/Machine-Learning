import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import scipy.io as scio
from sklearn.cluster import SpectralClustering

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

for gamma in [0.0001, 0.001 , 0.01, 0.1, 1, 10]:
    for clusters in [10,30,100]:
        classifierFunc = SpectralClustering(n_clusters=clusters, gamma=gamma)
        predLabels = classifierFunc.fit_predict(X)
        err = 0
        for i in range(len(predLabels)):
            if predLabels[i] != Y[i]:
                err += 1
        print("With gamma = %f and K = %f: Error ratio: %f" % (gamma, clusters , float(err) / len(predLabels)))
         # Plotting the graph
        plt.style.use("classic")

        colors = ['b', 'lime', 'c', 'r', 'y', 'm', 'k', 'teal', 'silver', 'aqua']
        figure, axis = plt.subplots()
        for i in range(10):
            nodes = np.array([X[j] for j in range(len(X)) if predLabels[j] == i])
            axis.scatter(nodes[:, 0], nodes[:, 1], s=16, c=colors[i])
        plt.show()
    