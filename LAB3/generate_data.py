from numpy import *
import numpy.random as rd
import matplotlib.pyplot as plt

def generate_data(feature, sample, center, cluster_mean, cluster_std):
    X = []
    for f in range(feature):
        for c in range(center):
            X.extend(rd.normal(cluster_mean[c][f],
                               cluster_std[c][f], int(sample/center)))
    X = mat(X).reshape(feature, sample).T
    return X