from pca import PCA
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import scipy.io as scio
from kmeans import KMeans

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

# apply KMeans with k = 10
centers, predictedLabels = KMeans(X.T, 10, 10)
# Error
err = 0
for i in range(len(predictedLabels)):
    if predictedLabels[i] != Y[i]:
        err += 1
print("Clustering Error ratio with Kmeans: ", float(err) / len(predictedLabels))

# PCA with d = 2 and d = 100
pcaarray = [2,100]
for i in pcaarray:
    print("For pca with dimensions = " , i)
    X = PCA(X.T, i)[-1].T
    
    # Plotting the graph
    plt.style.use("classic")

    colors = ['b', 'lime', 'c', 'r', 'y', 'm', 'k', 'teal', 'silver', 'aqua']
    figure, axis = plt.subplots()
    for i in range(10):
        nodes = np.array([X[j] for j in range(len(X)) if predictedLabels[j] == i])
        axis.scatter(nodes[:, 0], nodes[:, 1], s=16, c=colors[i])
    plt.show()