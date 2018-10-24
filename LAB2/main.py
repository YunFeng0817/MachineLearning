from numpy import *
import numpy.random as rd
import matplotlib.pyplot as plt
import math

DEGREE = 2
DENSITY = 30
X = []
Y = []
for i in range(1, DEGREE+1):
    X.extend(rd.normal(0, 20, int(DENSITY/2)))
    X.extend(rd.normal(50, 20, int(DENSITY/2)))
    # X.extend(rd.normal(0, i, int(DENSITY/2)))
    # X.extend(rd.normal(1, i, int(DENSITY/2)))
Y.extend(zeros(int(DENSITY/2)))
Y.extend(ones(int(DENSITY/2)))
X = matrix(X).reshape(DEGREE, DENSITY).T
Y = matrix(Y).reshape(DENSITY, 1)


def plot(X, w):
    fig1 = plt.figure('fig1')
    plt.plot(X[:int(DENSITY/2):, 0], X[:int(DENSITY/2):, 1], 'ro')
    plt.plot(X[int(DENSITY/2)::, 0], X[int(DENSITY/2)::, 1], 'go')
    start = -50
    end = 100
    x = arange(start, end, (end-start) /
               DENSITY).reshape(1, DENSITY)
    y = (w[0]+w[1]*x)/(-w[2])
    plt.plot(x, y, 'bo')
    plt.show(fig1)


def Gradient(X, Y, w):
    result = matrix(zeros(DEGREE+1)).reshape(DEGREE+1, 1)
    for i in range(DEGREE+1):
        sum = 0
        for j in range(DENSITY):
            x = w[0,0]
            temp = w[1::].T
            x += (temp) * (X[j, :].T)
            x = exp(x)
            x = 1 - 1/(1 + x)
            if i == 0:
                sum += Y[j]-x
            else:
                sum += X[j, i-1] * (Y[j]-x)
        result[i] = sum
    return result


w = matrix(zeros(DEGREE+1)).reshape(DEGREE+1, 1)
step = 1
i = 0
min_step_length = 1e-8
max_iteration = 1000
gamma = 1e-1
while ((step > min_step_length) and (i < max_iteration)):
    prev_position = w
    temp = Gradient(X, Y, w)
    w = w + gamma * temp
    step = (w - prev_position).T*(w - prev_position)
    i = i + 1
print(w, i)
plot(X, w)
