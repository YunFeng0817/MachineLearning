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
    return transpose(X)


def generateY(x):
    """
    return y with gauss noise according to argument x
    """
    return func(x) + rd.normal(0, 0.01, x.shape[0])


def analytical(x, y):
    """
    get theta by analytical solution
    """
    return linalg.inv(transpose(x)@x)@transpose(x)@y


def plot(x1, y1, x2, theta):
    """
    paint the plot
    """
    fig1 = plt.figure('fig1')
    plt.plot(x1, y1)
    X = generateX(x2, theta.size-1)
    plt.plot(x2, X @ theta)
    plt.show(fig1)


DEGREE = 10
START = 0
END = 5
STEP = 0.1
RESULT_STEP = 0.001  # step length for  painting the fit result
alpha = 1e-5
m = 0.2
x = arange(START, END, STEP)
X = generateX(x, DEGREE)
Y = generateY(x).reshape(x.size, 1)
# alpha = array([1e-4, 1e-4, 1e-4, 1e-7,1e-7]).reshape(DEGREE+1, 1)  # step length
# m = array([6e-1, 6e-1, 6e-1, 6e-1,6e-1]).reshape(DEGREE+1, 1)

# def destination(theta, x, y):
#     d = x@theta-y
#     return (1/2/x.size)*transpose(d)@(d)


# def Gradient(theta, x, y):
#     d = x@theta-y
#     return (1/x.size)*transpose(x)@d


# theta = zeros(DEGREE+1).reshape(DEGREE+1, 1)
# v = zeros(DEGREE+1).reshape(DEGREE+1, 1)
# gradient = Gradient(theta, X, Y)
# while not all(absolute(gradient) < 1e-3):
#     if all(absolute(gradient) > 1e10):
#         break
#     v = - alpha * gradient + m * v
#     theta = theta + v
#     gradient = Gradient(theta, X, Y)
#     # print(gradient)

theta = analytical(X, Y)

plot(x, Y, arange(START, END, RESULT_STEP), theta)
