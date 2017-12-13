import numpy as np

def dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


def computeRandomCentroids(data, k):
    idx = np.random.choice(np.arange(data.shape[0]), size=k, replace=False)
    return data[idx]

def getLabels(data, centers):
    labels = []
    for i, p in enumerate(data):
        dists = dist(p, centers)
        labels.append(np.argmin(dists))
    return labels



def computeError(data, centers, labels):
    errFound = 0
    for i, p in enumerate(data):
        errFound += dist(p, centers[labels[i]], axis=0)
    return errFound


def computeCentroids(data, labels, k):
    centroids = []
    for i in range(k):
        points = [data[j] for j in range(len(data)) if labels[j] == i]
        centroids.append(np.mean(points, axis=0))
    return centroids



def KMeans(data,cluster_count,iterations):
    data = data.T
    err, pred_centers, pred_labels = float("inf"), None, None
    for i in range(iterations):
        labels = np.zeros(len(data))
        centers = computeRandomCentroids(data,cluster_count)
        while True:
            old_labels = np.copy(labels)
            labels = getLabels(data, centers)
            if np.array_equal(labels, old_labels):
                errFound = computeError(data, centers, labels)
                if errFound < err:
                    err, pred_centers, pred_labels = errFound, centers, labels
                break
            centers = computeCentroids(data, labels,cluster_count)
    return pred_centers, pred_labels