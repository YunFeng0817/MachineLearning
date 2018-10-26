from numpy import *
import numpy.random as rd
import matplotlib.pyplot as plt
import math

DEGREE = 2  # the degree of the X
DENSITY = 500  # number of the generated data
type1_mean = 0  # mean of type1's data
type2_mean = 10  # mean of type2's data
type1_variance = 8  # variance of type1's data
type2_variance = 8  # variance of type2's data


def generate_data(scale):
    '''
    use gassus distribution to generate data
    hypothesis:
    each degree of X is independent
    each degree of X has same variance
    '''
    X = []
    Y = []
    for i in range(1, DEGREE+1):
        X.extend(rd.normal(type1_mean, type1_variance, int(scale/2)))
        X.extend(rd.normal(type2_mean, type2_variance, int(scale/2)))
    Y.extend(zeros(int(scale/2)))
    Y.extend(ones(int(scale/2)))
    X = mat(X).reshape(DEGREE, scale).T
    Y = mat(Y).reshape(scale, 1)
    return (X, Y)


def plot(X, w, **kwargs):
    '''
    draw the data set and the classification result
    '''
    if 'figure' in kwargs:
        plt.figure(kwargs['figure'])
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
    plt.title(kwargs['title']+'  ' + kwargs['text'])
    plt.xlabel('Degree 1')
    plt.ylabel('Degree 2')
    plt.draw()


def possible(X, w):
    '''
    compute the possibility of the P(Yi=1|Xi,w)
    '''
    Ones = mat(ones(DENSITY)).reshape(DENSITY, 1)  # to be easier to process w0
    newX = c_[Ones, X]
    x = newX * w
    x = exp(x)
    return 1-1/(1+x)


def accuracy(X, Y, w):
    Ones = mat(ones(X.shape[0])).reshape(
        X.shape[0], 1)  # to be easier to process w0
    newX = c_[Ones, X]
    result = newX*w
    result[where(result >= 0)] = 1
    result[where(result < 0)] = 0
    return 1-sum(logical_xor(Y, result))/X.shape[0]


def Gradient(X, Y, w):
    '''
    compute the gradient in current position according the w,X,Y
    '''
    Ones = mat(ones(DENSITY)).reshape(DENSITY, 1)  # to be easier to process w0
    newX = c_[Ones, X]
    return newX.T * (Y-possible(X, w))


def gradient_ascent(X, Y, reg):
    '''
    use gradient ascent method to compute the logistic regression
    if reg is false :
        use gradient ascent without regular term
    if reg is true :
        use gradient ascent with regular term
    '''
    w = mat(zeros(DEGREE+1)).reshape(DEGREE+1, 1)
    step = 1
    i = 0
    min_step_length = 1e-3
    max_iteration = 1e3
    gamma = 1e-3
    Lambda = -5
    while ((step > min_step_length) and (i < max_iteration)):
        prev_position = w
        w = w + gamma * Gradient(X, Y, w) - gamma*Lambda*reg*w
        step = (w - prev_position).T*(w - prev_position)
        i = i + 1
    print(w, i)
    return w


def newton_item(X, Y, w, reg):
    '''
    compute newton method's item:
    (hessian)-1 * (first derivative)
    if reg is false :
        use newton method without regular term
    if reg is true :
        use newton method with regular term
    '''
    Lambda = -5
    Ones = mat(ones(DENSITY)).reshape(DENSITY, 1)  # to be easier to process w0
    newX = c_[Ones, X]
    possible_result = possible(X, w)
    temp = multiply(possible_result, 1-possible_result)
    A = multiply(identity(DENSITY), temp)
    hessian = newX.T*A*newX
    return hessian.I*(Gradient(X, Y, w)-Lambda*reg*w)


def newton(X, Y, reg):
    '''
    use newton method to compute the logistic regression
    if reg is false :
        use newton method without regular term
    if reg is true :
        use newton method with regular term
    '''
    w = mat(zeros(DEGREE+1)).reshape(DEGREE+1, 1)
    step = 1
    i = 0
    min_step_length = 1e-3
    max_iteration = 10000
    gamma = 1e-1
    while ((step > min_step_length) and (i < max_iteration)):
        prev_position = w
        try:
            w = w + gamma * newton_item(X, Y, w, reg)
        except:
            print('caculate inverse matrix of Hassian error')
            break  # the hessian doesn't has inverse matrix
        step = (w - prev_position).T*(w - prev_position)
        i = i + 1
    print(w, i)
    return w

# generate the data for training
train_data = generate_data(DENSITY)
train_X = train_data[0]
train_Y = train_data[1]
# generate the the data for test
test_data = generate_data(int(0.4*DENSITY))
test_X = test_data[0]
test_Y = test_data[1]

w = gradient_ascent(train_X, train_Y, False)
plot(train_X, w, figure=1, title='gradient ascent without regular term',
     text='accuracy = '+str(accuracy(test_X, test_Y, w)))

w = gradient_ascent(train_X, train_Y, True)
plot(train_X, w, figure=2, title='gradient ascent with regular term',
     text='accuracy = '+str(accuracy(test_X, test_Y, w)))

w = newton(train_X, train_Y, False)
plot(train_X, w, figure=3, title='newton method without regular term',
     text='accuracy = '+str(accuracy(test_X, test_Y, w)))

w = newton(train_X, train_Y, True)
plot(train_X, w, figure=4, title='newton method with regular term',
     text='accuracy = '+str(accuracy(test_X, test_Y, w)))
     
plt.show()
