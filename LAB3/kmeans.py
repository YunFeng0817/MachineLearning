from numpy import *
import numpy.random as rd
import matplotlib.pyplot as plt
import generate_data
import math

# data property
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
max_iteration = int(1e5)
init_num = 10


def get_distance(x, y):
    return (x-y)*(x-y).T


def kmeans(X, center, max_iteration):
    sample = X.shape[0]
    feature = X.shape[1]
    # generate center point by random
    center_vector = mat([])
    for c in range(center):
        if center_vector.size == 0:
            center_vector = X[round(rd.rand()*(sample-1))]
        else:
            center_vector = r_[center_vector, X[round(rd.rand()*(sample-1))]]

    # initial the cluster
    cluster = mat(zeros(sample)).reshape(sample, 1)
    # iterate to get better cluster
    for i in range(max_iteration):
        pre_center = center_vector
        for s in range(sample):
            min_distance = Inf
            min_index = 0
            for c in range(center):
                distance = get_distance(X[s], center_vector[c])
                if min_distance > distance:
                    min_distance = distance
                    min_index = c
            cluster[s] = min_index

        # compute new center vector
        for c in range(center):
            cluster_points = X[where(cluster == c), :][0]
            for f in range(feature):
                center_vector[c, f] = sum(
                    cluster_points[:, f])/cluster_points.shape[0]

        # if the center vector doesn't change during this iteration,stop iterate
        if all(pre_center - center_vector) == 0:
            break
    cluster_num = []
    for c in range(center):
        cluster_num.append(X[where(cluster == c), :][0].shape[0])
    return (center_vector, cluster, cluster_num)


def my_plot(X, center_vector, cluster, **kwargs):
    """
    X: the original data
    center_vector: the vector of the center point of each cluster
    cluster: vector means which cluster the sample X[n,:] belong to
    kwargs: figure: the name of the painted figure
    """
    center = center_vector.shape[0]
    if 'figure' in kwargs:
        plt.figure(kwargs['figure'])
    else:
        plt.figure(1)
    # draw the standard cluster
    plt.subplot(1, 2, 1)
    count = 0
    for c in range(center):
        plt.scatter(X[count:int(count+X.shape[0]/center), 0].tolist(),
                    X[count:int(count+X.shape[0]/center), 1].tolist())
        count += int(X.shape[0]/center)
    # draw the cluster result
    plt.subplot(1, 2, 2)
    for c in range(center):
        cluster_points = X[where(cluster == c), :][0]
        plt.scatter(cluster_points[:, 0].tolist(),
                    cluster_points[:, 1].tolist())
        plt.scatter(center_vector[c, 0].tolist(), center_vector[c, 1].tolist())
    plt.draw()


if __name__ == "__main__":
    X = generate_data.generate_data(
        FEATURE, SAMPLE, CENTER, CLUSTER_MEAN, CLUSTER_STD)
    result = kmeans(X, CENTER, max_iteration)
    my_plot(X, result[0], result[1])
    print(result[0])
    print(result[2])
    plt.show()
