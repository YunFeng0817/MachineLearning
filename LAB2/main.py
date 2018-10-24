from numpy import *
import numpy.random as rd
import matplotlib.pyplot as plt
import math

DEGREE = 2  # the degree of the X
DENSITY = 50  # number of the generated data
type1_mean = 0  # mean of type1's data
type2_mean = 1  # mean of type2's data
type1_variance = 0.5  # variance of type1's data
type2_variance = 0.5  # variance of type2's data


def generate_data():
    '''
    use gassus distribution to generate data
    hypothesis:
    each degree of X is independent
    each degree of X has same variance
    '''
    X = []
    Y = []
    for i in range(1, DEGREE+1):
        X.extend(rd.normal(type1_mean, type1_variance, int(DENSITY/2)))
        X.extend(rd.normal(type2_mean, type2_variance, int(DENSITY/2)))
    Y.extend(zeros(int(DENSITY/2)))
    Y.extend(ones(int(DENSITY/2)))
    X = mat(X).reshape(DEGREE, DENSITY).T
    Y = mat(Y).reshape(DENSITY, 1)
    return (X, Y)


def plot(X, w):
    '''
    draw the data set and the classification result
    '''
    plt.scatter(X[:int(DENSITY/2):, 0].tolist(),
                X[:int(DENSITY/2):, 1].tolist(), label='class1')
    plt.scatter(X[int(DENSITY/2)::, 0].tolist(),
                X[int(DENSITY/2)::, 1].tolist(), label='class2')
    start = type1_mean-2*type1_variance
    end = type2_mean+2*type2_variance
    x = linspace(start, end, 100)
    w = w.tolist()
    y = -(w[0]+w[1]*x)/w[2]
    plt.plot(x, y)
    plt.show()


def Gradient(X, Y, w):
    '''
    compute the gradient in current position according the w,X,Y
    '''
    result = mat(zeros(DEGREE+1)).reshape(DEGREE+1, 1)
    for i in range(DEGREE+1):
        sum = 0
        for j in range(DENSITY):
            x = w[0, 0]
            x += (w[1::].T) * (X[j, :].T)
            x = exp(x)
            x = 1 - 1/(1 + x)
            if i == 0:
                sum += Y[j]-x
            else:
                sum += X[j, i-1] * (Y[j]-x)
        result[i] = sum
    return result


def gradient_discent(X, Y):
    '''
    use gradient descent method to compute the logistic regression
    '''
    w = mat(zeros(DEGREE+1)).reshape(DEGREE+1, 1)
    step = 1
    i = 0
    min_step_length = 1e-3
    max_iteration = 5000
    gamma = 1e-1
    while ((step > min_step_length) and (i < max_iteration)):
        prev_position = w
        w = w + gamma * Gradient(X, Y, w)
        step = (w - prev_position).T*(w - prev_position)
        i = i + 1
    print(w, i)
    return w


data = generate_data()
X = data[0]
Y = data[1]
w = gradient_discent(X, Y)
plot(X, w)
