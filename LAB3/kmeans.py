from numpy import *
import numpy.random as rd
import matplotlib.pyplot as plt
import math

# data property
FEATURE = 2
SAMPLE = 10
DATA_CENTER = 2
CLUSTER_MEAN = [[20, 40], [70, 10]]
CLUSTER_STD = [1, 2]

# algorithm property
CENTER = 2
max_iteration = 100
init_num = 10


def generate_data(feature, sample, center, cluster_mean, cluster_std):
    X = []
    for f in range(feature):
        for c in range(center):
            X.extend(rd.normal(cluster_mean[c][f],
                               cluster_std[c], int(sample/center)))
    X = mat(X).reshape(feature, sample).T
    return X


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
                center_vector[c, f] = sum(cluster_points[:, f])/center

        # if the center vector doesn't change during this iteration,stop iterate
        if all(pre_center - center_vector) == 0:
            break
    return (center_vector, cluster)


X = generate_data(FEATURE, SAMPLE, CENTER, CLUSTER_MEAN, CLUSTER_STD)
kmeans(X, CENTER, max_iteration)
