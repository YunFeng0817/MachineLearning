from numpy import *
import numpy.random as rd
import matplotlib.pyplot as plt
import generate_data
from kmeans import my_plot

FEATURE = 2
SAMPLE = 1000
DATA_CENTER = 4
CLUSTER_MEAN = [
    [100, 10],
    [200, 300],
    [50, 400],
    [300, 500]
]
CLUSTER_STD = [
    [20, 10],
    [30, 30],
    [10, 10],
    [90, 50]
]

# algorithm property
CENTER = DATA_CENTER

X = generate_data.generate_data(
    FEATURE, SAMPLE, CENTER, CLUSTER_MEAN, CLUSTER_STD)


def GaussPossible(X, U, conv, k, n):
    """
    X: the original data
    U: the parameter in EM algorithm , represent the mean value
    conv: the parameter in EM algorith, represent the covariance matrix
    k: represent the number of clusters
    n: represent the nth sample in data X
    """
    conv = mat(conv)
    degree = X.shape[0]
    conv_value = linalg.det(conv)
    return 1/(power(2*pi, degree/2)*power(abs(conv_value), 0.5))*exp(-(X[:, n]-U[k, :].T).T*conv.I*(X[:, n]-U[k, :].T)/2)


def GMM(K, X):
    """
    K: represent the number of clusters
    X: the original data
    """
    # init the w
    ERROR = 1e-2
    w = mat(ones(K)).reshape(K, 1)
    w = w/sum(w)
    U = mat(rd.randint(int(X.min()), int(X.max()),
                       K*X.shape[0])).reshape(K, X.shape[0])
    conv = [0]*K
    log_likehood = mat(zeros(X.shape[1])).reshape(X.shape[1], 1)
    old_likehood = mat(ones(X.shape[1])).reshape(X.shape[1], 1)
    # init the convs with the covariance of whole data
    for i in range(K):
        conv[i] = cov(X)
    count = 0
    while((log_likehood-old_likehood).T*(log_likehood-old_likehood) > ERROR):
        count += 1
        old_likehood = log_likehood
        G = mat(zeros(K*X.shape[1])).reshape(K, X.shape[1])
        for i in range(K):
            for j in range(X.shape[1]):
                G[i, j] = GaussPossible(X, U, conv[i], i, j)
        # E step
        gamma = divide(multiply(w, G), w.T*G)

        # M step
        Nk = sum(gamma, 1)
        Nk_I = 1/Nk
        U = multiply(Nk_I, gamma*X.T)
        for i in range(K):
            Sum = mat(zeros(X.shape[0]*X.shape[0])
                      ).reshape(X.shape[0], X.shape[0])
            for j in range(X.shape[1]):
                Sum += gamma[i, j]*(X[:, j]-U[i, :].T)*(X[:, j]-U[i, :].T).T
            conv[i] = float(Nk_I[i])*Sum
        w = Nk/X.shape[1]
        log_likehood = log((w.T*G).T)
    print("Loop times : "+str(count))
    cluster = []
    for i in range(X.shape[1]):
        possible = []
        for j in range(K):
            possible.append(float(GaussPossible(X, U, conv[j], j, i)))
        possible = mat(possible).reshape(len(possible), 1)
        cluster.append(where(possible[:, 0] == possible.max())[0])
    return (U, mat(cluster))


result = GMM(DATA_CENTER, X.T)
my_plot(X, result[0], result[1]) # use the plot function in kmeans.py
plt.show()
