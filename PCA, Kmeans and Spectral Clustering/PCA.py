from scipy.io import loadmat
import scipy.io
import numpy as np
np.set_printoptions(threshold=np.nan)
from matplotlib import pyplot as plotter
import os
import ntpath

inputData = dict()
inputPath = "HW3_Data/dataset1.mat"
inputPath2 = "HW3_Data/dataset2.mat"

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# compute D(40) dimesional mean vector by first computing cov matrix
# calculate eigen vectors
# sort the eigen vectors in decreasing ordera and choose 2 highest value vectors: 
# Use the eigen vector matrix to transform the samples onto the new subspace

def calculate_pca(inputPath):
    inputData = loadmat(inputPath)
    pcaData = inputData['Y']
    print("First Feature Vs. Second Feature...........")
    plotCluster(pcaData, 0, 1)
    #plotter.savefig(os.path.join(os.getcwd(), path_leaf(inputPath) + ' Feat1 vs 2.png'))
    print("Second Feature Vs. Third Feature...........")
    plotCluster(pcaData, 1, 2)
    #plotter.savefig(os.path.join(os.getcwd(), path_leaf(inputPath) + 'Feat2 vs 3.png'))
    vector_Mean = pcaData.mean(axis=1, keepdims=True)
    print("MeanVector is",vector_Mean)
    pcaData = pcaData - vector_Mean
    U, S, V = np.linalg.svd(pcaData)
    U = U[:, [0,1]]
    X = U.T.dot(pcaData)
    print()
    print("2-dimensional representation of data using PCA is")
    print(X)
    arr = X.reshape((2,200))  # 2d array of 2X200
    scipy.io.savemat('HW3_Data/2D'+ path_leaf(inputPath), mdict={'arr': arr})
    print("Plot of reduced vector usin PCA....")
    plotCluster(X, 0, 1)
    return vector_Mean, U, X

  
    #plotter.savefig(os.path.join(os.getcwd(), path_leaf(inputPath) + ' PCAPlot.png'))
    
def rbf_kernel():
    inputData = loadmat(inputPath)
    kernelData = inputData['Y']
    dat = kernelData.shape[1]
    krnl = np.zeros((dat, dat))
    for i in range(dat):
        for j in range(dat):
            krnl[i, j] = np.sum((kernelData[:, i] - kernelData[:, j]) ** 2) / 2
    return np.exp(krnl)
 
    
def calculate_kpca(K, d):
    lengt = len(K)
    onesmat = np.ones((lengt, lengt)) / lengt
    revisedK = K - K.dot(onesmat) - onesmat.dot(K) + onesmat.dot(K).dot(onesmat)
    w,v = np.linalg.eig(revisedK)
    pairs = sorted([(w[i], v[:, i]) for i in range(len(w))], key=lambda x: x[0], reverse=True)
    finData =  K.dot(np.column_stack((pairs[i][1] for i in range(d)))).T
    print()
    print("2-dimensional representation of data using Kernel PCA is")
    print(X)
    plotCluster(finData, 0, 1)

# Getting the values and plotting it
def plotCluster(Y, x1, x2):
    f1 = np.ravel(Y[x1, :])
    f2 = np.ravel(Y[x2, :])
    plotter.xlabel("dimension 1")
    plotter.ylabel("dimension 2")
    plotter.scatter(f1, f2, c='blue', s=7)
    plotter.show()
       
    
mu, U, X = calculate_pca(os.path.splitext(inputPath)[0])
mu, U, X = calculate_pca(os.path.splitext(inputPath2)[0])
kernel = rbf_kernel()
calculate_kpca(kernel,2)

