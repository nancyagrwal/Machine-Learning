from copy import deepcopy
from scipy.linalg import norm
from scipy.io import loadmat
from scipy import array, mean, rand, ones
import os
import ntpath
from matplotlib import pyplot as plotter

def calculate_center_cost(centroids, centroids_dict):
    count = 0
    euc_dis = 0

    for v in centroids_dict:
        cen = centroids[:,v]
        for datapoint in centroids_dict[v]:
            count += 1;euc_dis += norm(cen - datapoint) ** 2
    final_cost = euc_dis / count
    return final_cost


def ClusterKmeansData(kmeansData, cluster_count, iterations):
    minimum_cost = MAX_VALUE
    center_to_data_point = None

    for i in range(iterations):
        exist_centers = rand(kmeansData.shape[0], cluster_count)
        centroids_dict = dict(); centroids_dictIndex = dict()
        iter_max = 1000

        for i in range(iter_max):
            centroids_dict = dict()
            centroids_dictIndex = dict()
            clusterd = ones(kmeansData.shape[1]) * -1

            # Compute the centroid for each data point
            for indexs in range(kmeansData.shape[1]):
                pointData = kmeansData[:, indexs]
                index_of_closest_center = -1
                closest_distance = MAX_VALUE

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

            center_new = deepcopy(exist_centers)
            for s in range(exist_centers.shape[1]):
                if s in centroids_dict:center_new[:, s] = mean(array(centroids_dict[s]).transpose(), 1)
            if (exist_centers == center_new).all():break
            else:exist_centers = center_new
        cost = calculate_center_cost(exist_centers, centroids_dict)

        # Choose the best Cluster based on Cost function
        if minimum_cost > cost:center_to_data_point = centroids_dictIndex
    return center_to_data_point

# perform clustering
def returnClusteredData(centroids_dict):
    finData = []
    for h in centroids_dict:
        for m in centroids_dict[h]:finData.append((h,m))
    sorteddata = sorted(finData, key = lambda x:x[1])
    finData = list(sorteddata)
    finData = list(map(lambda x:x[0], finData))
    return finData

def original_plot(Y,path):
    xl = [min(Y[0, :]) , max(Y[0, :])]
    yl = [min(Y[1, :]) , max(Y[1, :])]
    plotter.xlim(xl); plotter.ylim(yl)
    plotter.plot(Y[0, :], Y[1, :], 'o')
    plotter.xlim(xl)
    plotter.ylim(yl)
    plotter.savefig(os.path.join(os.getcwd(), path+ ' original data.png'))
    plotter.show()
    
# Plotting Graph after Kmeans Clustering
def cluster_plot(cdp, kmeansData,path):
    cendic = dict()
    xl = [min(kmeansData[0, :]) , max(kmeansData[0, :])]
    yl = [min(kmeansData[1, :]) , max(kmeansData[1, :])]
    plotter.xlim(xl); plotter.ylim(yl)
    for cen_ind in cdp:
        cendic[cen_ind] = list()
        for i in cdp[cen_ind]:
            cendic[cen_ind].append(kmeansData[:, i])

    for cen_ind in cendic:
        centroidData = array(cendic[cen_ind]).transpose()
        plotter.plot(centroidData[0, :], centroidData[1, :], 'o')
    
    plotter.savefig(os.path.join(os.getcwd(), path +' clustered data.png'))
    plotter.show()

  
    
def find_k_means(kmeansData, cluster_count, max_iterations,path):
    data_centroids = ClusterKmeansData(kmeansData, cluster_count, max_iterations)
    finData = returnClusteredData(data_centroids)
    print ("Indices of points in each group")
    print(finData)
    original_plot(kmeansData,path)
    cluster_plot(data_centroids, kmeansData,path)
    
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

inputPath = "HW3_Data/dataset1.mat"
inputPath2D = "HW3_Data/2Ddataset1.mat"
inputPath2 = "HW3_Data/2Ddataset2.mat"
MAX_VALUE = float('inf')
MIN_VALUE = -float('inf')


dataDict = loadmat(os.path.join(os.getcwd(), inputPath))
kmeansData = dataDict['Y']
find_k_means(kmeansData, 2, 10,path_leaf(os.path.splitext(inputPath)[0]))

dataDict2 = loadmat(os.path.join(os.getcwd(), inputPath2D))
kmeansData2 = dataDict2['arr']
find_k_means(kmeansData2, 2, 10,path_leaf(os.path.splitext(inputPath2D)[0]))

dataDict2 = loadmat(os.path.join(os.getcwd(), inputPath2))
kmeansData2 = dataDict2['arr']
find_k_means(kmeansData2, 2, 10,path_leaf(os.path.splitext(inputPath2)[0]))



