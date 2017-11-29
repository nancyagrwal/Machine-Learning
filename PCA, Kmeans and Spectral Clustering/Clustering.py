import os
from copy import deepcopy
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plotter
from scipy import mean, rand, array,  exp, zeros, ones
from numpy.linalg import eig, norm
from sklearn.preprocessing import MinMaxScaler

class Q6:
    def __init__(self):
        self.dataDict = dict()
        self.MAX_VALUE = float('inf')
        self.MIN_VALUE = -float('inf')

        inputPath = "HW3_Data/dataset2.mat"
        self.dataDict = loadmat(os.path.join(os.getcwd(), inputPath))
        self.spectralData = self.dataDict['Y']
    
   
    def KPCA(self,d,sigma):
      
        Y = self.spectralData
        N = Y.shape[1]
        kernel = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                kernel[i, j] = np.sum((Y[:, i] - Y[:, j]) ** 2) / sigma
        K = np.exp(kernel) 
        lengt = len(K)
        onesmat = np.ones((lengt, lengt)) / lengt
        revisedK = K - K.dot(onesmat) - onesmat.dot(K) + onesmat.dot(K).dot(onesmat)
        w,v = np.linalg.eig(revisedK)
        pairs = sorted([(w[i], v[:, i]) for i in range(len(w))], key=lambda x: x[0], reverse=True)
        finData =  K.dot(np.column_stack((pairs[i][1] for i in range(d)))).T
        self.plotCluster(finData, 0, 1)
       
    
    def plotCluster(self,Y, x1, x2):
            f1 = np.ravel(Y[x1, :])
            f2 = np.ravel(Y[x2, :])
            plotter.xlabel("dimension 1")
            plotter.ylabel("dimension 2")
            plotter.scatter(f1, f2, c='blue', s=7)
            plotter.show()
            
    def spectral(self, k, sigma):
        dat = self.spectralData.shape[1];
        weight_matrix = ones(shape=(dat,dat), dtype=float)
        degree_matrix = zeros(shape=(dat,dat), dtype=float)
         
        for m in range(weight_matrix.shape[0]):
            for n in range(weight_matrix.shape[1]):
                dat1 = self.spectralData[:,m] - self.spectralData[:,n];
                weight_matrix[m][n] = exp(-1 * (norm(dat1) ** 2) / sigma)
        
        for l in range(weight_matrix.shape[0]):
            degree_matrix[l][l] = sum(weight_matrix[l, :])
        
        lap_diff = MinMaxScaler().fit_transform(degree_matrix - weight_matrix)

        # calculate the eigen values and eigen vectors and select bottom k:
        eigen_val, eigen_vec = eig(lap_diff); eigen_vec = eigen_vec.real
        indexes = eigen_val.argsort()
        eigen_vec = MinMaxScaler().fit_transform(eigen_vec)
        
        lower_k_vec = indexes[:k]
        trans = eigen_vec[:,lower_k_vec].transpose()
        self.plotCluster(trans, 0, 1)
        
        
    def find_k_mean(self, cluster_count, iterations, spectralData):
        print(spectralData.shape)
        min_cost = self.MAX_VALUE
        center_to_data_point = None

        for i in range(iterations):
            exist_centers = rand(spectralData.shape[0], cluster_count)
            centroids_dict = dict(); centroids_dictIndex = dict()
            iter_max = 1000
     
            for i in range(iter_max):
                centroids_dict = dict()
                centroids_dictIndex = dict()
                clusterd = ones(spectralData.shape[1]) * -1
               
                # Update the centroids for data points
                for indexs in range(spectralData.shape[1]):
                    pointData = spectralData[:, indexs]
                    index_of_closest_center = -1
                    closest_distance = self.MAX_VALUE
    
                    for s in range(exist_centers.shape[1]):
                        curr_cent = exist_centers[:, s]
                        euclidian_distance = 2 ** norm(pointData - curr_cent)
                        if closest_distance > euclidian_distance:
                            closest_distance = euclidian_distance
                            index_of_closest_center = s
                    clusterd[indexs] = index_of_closest_center
                    if index_of_closest_center not in centroids_dict:centroids_dict[index_of_closest_center] = list()
                    if index_of_closest_center not in centroids_dictIndex:centroids_dictIndex[index_of_closest_center] = list()
                    centroids_dict[index_of_closest_center].append(pointData)
                    centroids_dictIndex[index_of_closest_center].append(indexs)
                    
                # Update the centroids based on data points
                center_new = deepcopy(exist_centers)
                for s in range(exist_centers.shape[1]):
                    if s in centroids_dict:center_new[:, s] = mean(array(centroids_dict[s]).transpose(), 1)
                if (exist_centers == center_new).all():break
                else:exist_centers = center_new
                
            cost = self.calculate_center_cost(exist_centers, centroids_dict)
          
            if min_cost > cost:center_to_data_point = centroids_dictIndex
        return center_to_data_point

    def calculate_center_cost(self,cenlist, centroids_dict):
        count = 0
        euc_dis = 0
    
        for v in centroids_dict:
            cen = cenlist[:,v]
            for i in centroids_dict[v]:
                count += 1;euc_dis += norm(cen - i) ** 2
        final_cost = euc_dis / count
        return final_cost  

  
    def plot_clustered_data(self, center_to_pts, xl, yl):
        centroids_dict = dict()
        print ("Indices of points in each group")
        print(center_to_pts)
        
        for index_of_center in center_to_pts:
            centroids_dict[index_of_center] = list()
            for index_of_dp in center_to_pts[index_of_center]:
                centroids_dict[index_of_center].append(self.spectralData[:, index_of_dp])

        for index_of_center in centroids_dict:
            dt_pts_centr = array(centroids_dict[index_of_center]).transpose()
            plotter.plot(dt_pts_centr[0, :], dt_pts_centr[1, :], 'o')
        plotter.xlim(xl); plotter.ylim(yl)
        #plotter.savefig(os.path.join(os.getcwd(), 'clustered_graph.png'))
        plotter.show()


spectral = Q6()

sigmas = [8, 4, 2, 1]
for j in range(len(sigmas)): spectral.KPCA(2,sigmas[j])
for j in range(len(sigmas)): spectral.spectral(2,sigmas[j])

