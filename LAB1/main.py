from numpy import *
import numpy.random as rd
import matplotlib.pyplot as plt
import math


def func(x):
    """
    original function whitch to be fitted
    """
    return sin(math.pi*x)


def generateX(x, degree):
    """
    return vedemend matrix
    """
    X = ones(x.size)
    for i in range(degree):
        X = row_stack((X, x**(i+1)))
    X = mat(X)
    return X.T


def generateY(x, noise_mean, noise_variance):
    """
    return y with gauss noise according to argument x
    """
    return mat(func(x) + rd.normal(noise_mean, noise_variance, x.shape[0]))


def analytical(x, y, Lambda):
    """
    get theta by analytical solution
    """
    if Lambda == '':
        return (x.T*x).I*x.T*y
    else:
        return (x.T*x + Lambda).I*x.T*y


def grandient_descent(x, y, min_step_length, max_iteration, gamma, Lambda):
    """
    gradient descent solution
    arguments:
        x : origin data x
        y : origin data y
        min_step_length : descent min step
        max_iteration :　max times to iterate
        gamma : iteration step
        lamda : regular term parameter
    """
    def gradient(theta, x, y, Lambda):
        """
        compute the gradient
        arguments:
            x : origin data x
            y : origin data y
            lamda : regular term parameter; if lamda is '',means without regular term
        """
        if Ｌambda == '':
            return x.T * x * theta - x.T * y
        else:
            return x.T * x * theta - x.T * y+Lambda*theta

    theta = mat(zeros(DEGREE+1)).reshape(DEGREE+1, 1)
    v = zeros(DEGREE+1).reshape(DEGREE+1, 1)
    step = 1
    i = 0
    while ((step > min_step_length) and (i < max_iteration)):
        prev_position = theta
        theta = theta - gamma * gradient(theta, x, y, Lambda)
        step = (theta - prev_position).T*(theta - prev_position)
        i = i + 1
    return theta


def plot(x1, y1, x2, theta):
    """
    paint the plot

    arguments:
        x1: origin data x
        y1: origin data y
        x2: fit function x
        theta : ploynormial fit function's parameter
    """
    fig1 = plt.figure('fig1')
    plt.plot(x1, y1)
    X = generateX(x2, theta.size-1)
    plt.plot(x2, X @ theta)
    plt.show(fig1)


DEGREE = 10
START = -1  # origin data left interval
END = 1  # origin data right interval
ORIGIN_STEP = 0.04  # origin data step
RESULT_STEP = 0.001  # step length for  painting the fit result
MAX_ITERATION = 1e9  # parameter for gradient descent
MIN_STEP_LENGTH = 1e-9  # parameter for gradient descent
NOISE_MEAN = 0
NOISE_VARIANCE = 0.1
GAMMA = 1e-2
LAMBDA = 0.1  # regular term parameter
x = arange(START, END, ORIGIN_STEP)
X = generateX(x, DEGREE)
Y = generateY(x, NOISE_MEAN, NOISE_VARIANCE).reshape(x.size, 1)


# theta = analytical(X, Y, '')
# theta = analytical(X, Y, LAMBDA)
theta = grandient_descent(X, Y, MIN_STEP_LENGTH, MAX_ITERATION, GAMMA, '')
# theta = grandient_descent(X, Y, MIN_STEP_LENGTH, MAX_ITERATION, GAMMA, LAMBDA)


plot(x, Y, arange(START, END, RESULT_STEP), theta)
