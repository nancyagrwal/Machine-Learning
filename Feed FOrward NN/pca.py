import numpy as np

def PCA(data,dimension):
    """
    :param data: data matrix(D * N)
    :param dimension: dimension of the subspace
    :return: meanu: mean of the subspace(D)
             U: subspace basis(D * d)
             X: low-dimension representations(d * N)
    """
    meanu = data.mean(axis=1, keepdims=True)
    data = data - meanu
    # SV decomposition:
    U,S,V = np.linalg.svd(data)
    U = U[:,:dimension]
    X = U.T.dot(data)
    return meanu, U, X


def KPCA(Kernel,dimension):
    """
    :param Kernel: kernel matrix(N * N)
    :param dimension: dimension of the subspace
    :return: X: low-dimension representation of data(dimension * N)
    """
    N = len(Kernel)
    O = np.ones((N, N)) / N
    new_Kernel = Kernel - Kernel.dot(O) - O.dot(Kernel) + O.dot(Kernel).dot(O)
    eigen_val, eigen_vec = np.linalg.eig(new_Kernel)
    pairs = sorted([(eigen_val[i], eigen_vec[:, i]) for i in range(len(eigen_val))], key=lambda x: x[0], reverse=True)
    return Kernel.dot(np.column_stack((pairs[i][1] for i in range(dimension)))).T