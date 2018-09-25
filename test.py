import numpy as np
from numpy import *
import numpy.random as rd

# Size of the points dataset.
m = 20

def generate(RANGE, STEP):
    """
    Args:
        RANGE: int set the sample range between [0,RANGE]
        STEP: set the data by step length
    Returns:
        turple (x,y)
    """
    x = arange(0, RANGE, STEP)
    train_sample = sin(2*math.pi*x) + rd.normal(0, 0.001, x.shape[0])
    return (x, train_sample)


DEGREE = 8
RANGE = 50
STEP = 1
x = arange(0, RANGE, STEP)
X = ones(x.size)
for i in range(DEGREE):
    X = row_stack((X, x**(i+1)))
X = transpose(X)
y = generate(RANGE, STEP)[1].reshape(x.size, 1)
# alpha = array([1e-4, 1e-4, 1e-4, 1e-7,1e-7]).reshape(DEGREE+1, 1)  # step length
# m = array([6e-1, 6e-1, 6e-1, 6e-1,6e-1]).reshape(DEGREE+1, 1)
# The Learning Rate alpha.
alpha = 0.01

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
    '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), diff)

def gradient_descent(X, y, alpha):
    '''Perform gradient descent.'''
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta

optimal = gradient_descent(X, y, alpha)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])