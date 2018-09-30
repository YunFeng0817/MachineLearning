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
        return (x.T*x + Lambda * identity(DEGREE+1)).I*x.T*y


def loss(x, y, theta):
    return 0.5*(x*theta[0]-y).T*(x*theta[0]-y)


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


def conjugation_gradient(x, y, min_step_length, Lambda):
    if Lambda == '':
        A = x.T * x
    else:
        A = x.T * x + Lambda*identity(DEGREE+1)
    b = x.T * y
    theta = mat(zeros(DEGREE+1)).reshape(DEGREE+1, 1)

    r = b - A * theta
    p = r
    for i in range(DEGREE+1):
        a = (r.T * r)/(p.T*A*p)
        theta = theta + multiply(a, p)
        r_new = r - multiply(a, A*p)
        if r.T*r < min_step_length:
            break
        beta = (r_new.T * r_new)/(r.T*r)
        p = r_new + multiply(beta, p)
        r = r_new
    return theta


def plot(x1, y1, x2, theta, legend, xlabel, ylabel, title):
    """
    paint the plot

    arguments:
        x1: origin data x
        y1: origin data y
        x2: fit function x
        theta : ploynormial fit function's parameter
        legend : a list of every line's meaning
        xlabel : xlabel's meaning
        ylabel : ylabel's meaning
        title : the title of the plot
    """
    fig1 = plt.figure('fig1')
    plt.plot(x1, y1)
    for i in theta:
        X = generateX(x2, i.size-1)
        plt.plot(x2, X @ i)
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show(fig1)


DEGREE = 30
START = -1  # origin data left interval
END = 1  # origin data right interval
TRAIN_STEP = 0.02  # train data set step
VALID_STEP = 0.04  # valid data set step
RESULT_STEP = 0.001  # step length for  painting the fit result
MAX_ITERATION = 1e9  # parameter for gradient descent
MIN_STEP_LENGTH = 1e-9  # parameter for gradient descent
NOISE_MEAN = 0
NOISE_VARIANCE = 0.1
GAMMA = 1e-2
LAMBDA = 10  # regular term parameter

theta = []
legend = ['source data']
# for i in range(1, 4):
#     DEGREE = i
#     x = arange(START, END, TRAIN_STEP)
#     X = generateX(x, DEGREE)
#     Y = generateY(x, NOISE_MEAN, NOISE_VARIANCE).reshape(x.size, 1)
#     theta.append(analytical(X, Y, ''))
#     legend.append('Degree : ' + str(i))
legend.append('fit result')

# train
x = arange(START, END, TRAIN_STEP)
X = generateX(x, DEGREE)
Y = generateY(x, NOISE_MEAN, NOISE_VARIANCE).reshape(x.size, 1)
theta = [analytical(X, Y, '')] # analytical solution without regular term
theta = [analytical(X, Y, LAMBDA)] # analytical solution with regular term
theta = [grandient_descent(X, Y, MIN_STEP_LENGTH, MAX_ITERATION, GAMMA, '')] # gradient descent solution without regular term
theta = [grandient_descent(X, Y, MIN_STEP_LENGTH, MAX_ITERATION, GAMMA, LAMBDA)] # gradient descent solution with regular term
theta = [conjugation_gradient(X, Y, MIN_STEP_LENGTH, '')] # conjugate gradient solution without regular term
theta = [conjugation_gradient(X, Y, MIN_STEP_LENGTH, LAMBDA)] # conjugate gradient solution with regular term
# validation
x = arange(START, END, VALID_STEP)
X = generateX(x, DEGREE)
Y = generateY(x, NOISE_MEAN, NOISE_VARIANCE).reshape(x.size, 1)
loss = float(loss(X, Y, theta))
plot(x, Y, arange(START, END, RESULT_STEP), theta,
     legend, 'x', 'y', 'loss : '+str(loss))
