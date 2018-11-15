from numpy import *
import numpy.random as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
import operator


def PCA(X, K):
    """
    X : the original data  nxm (n-> number of the data set) (m-> data degree)
    K : the number of components needed
    """
    mean = sum(X, 0)/X.shape[0]
    X = X-mean  # normalize the original data X
    scatter_mat = dot(transpose(X), X)
    eig_value, eig_vec = linalg.eig(scatter_mat)
    # take the eigen value and eigenvector pair into one list
    eig_pair = [(abs(eig_value[i]), eig_vec[:, i]) for i in range(X.shape[1])]
    # print(eig_pair)
    # sort pair from big to small
    eig_pair.sort(key=lambda x: x[0], reverse=True)
    # select top K eigen vector
    chosen_pair = array([eig_vec[1]for eig_vec in eig_pair[:K]])
    return (dot(X, transpose(chosen_pair)), chosen_pair)


def generate_data(feature, sample, mean, variance):
    X = []
    for f in range(feature):
        X.extend(rd.normal(mean[f], variance[f], sample))
    X = mat(X).reshape(feature, sample).T
    return X


# data property
FEATURE = 3
SAMPLE = 1000
MEAN = [100, 30, 70]
VARIANCE = [10, 15, 0.1]
plt.figure(1)
X = generate_data(FEATURE, SAMPLE, MEAN, VARIANCE)
ax = plt.subplot(1, 3, 1, projection='3d')  # create a 3d project
ax.scatter(X[:, 0].tolist(), X[:, 1].tolist(), X[:, 2].tolist())
ax.set_xlim(X.min(), X.max())
ax.set_ylim(X.min(), X.max())
ax.set_zlim(X.min(), X.max())
result, main_mat = PCA(X, 2)
plt.subplot(1, 3, 2)
plt.axis([result.min(), result.max(), result.min(), result.max()])
plt.scatter(result[:, 0].tolist(), result[:, 1].tolist())
ax = plt.subplot(1, 3, 3, projection='3d')
restore = result*main_mat
ax.scatter(restore[:, 0].tolist(), restore[:,1].tolist(), restore[:, 2].tolist())

# 读取trainingMat
filename = './LAB4/train-images.idx3-ubyte'
binfile = open(filename, 'rb')
buf = binfile.read()
index = 0
# '>IIII'使用大端法读取四个unsigned int32
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')

# 读取labels
filename1 = './LAB4/train-labels.idx1-ubyte'
binfile1 = open(filename1, 'rb')
buf1 = binfile1.read()

index1 = 0
# '>IIII'使用大端法读取两个unsigned int32
magic1, numLabels1 = struct.unpack_from('>II', buf, index)
index1 += struct.calcsize('>II')


# 设置训练数目为2500个
trainingNumbers = 16
# 降维后的维度为７个维度　降维后的数据为40维度
K = 1
# 初始化traingMat
train_mat = zeros((trainingNumbers, 28*28))
CHOSEN_NUM = 3

plt.figure(2)
# 获取经过PCA  处理过的traingMat 和 label
i = 0
while i < trainingNumbers:
    numtemp = struct.unpack_from('1B', buf1, index1)
    label = numtemp[0]
    index1 += struct.calcsize('1B')
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')
    if label == CHOSEN_NUM:
        train_mat[i] = im
        image = array(im).reshape(28, 28)
        plt.subplot(12, 4, i+1)
        plt.imshow(image)
        i += 1


PCA_mat, main_mat = PCA(train_mat, K)

for i in range(trainingNumbers):
    im = dot(PCA_mat[i], main_mat)+sum(train_mat, 0)/train_mat.shape[0]
    image = array(im).reshape(28, 28).astype(int)
    plt.subplot(12, 4, i+17)
    plt.imshow(image)

# 读取testMat
filename3 = './LAB4/t10k-images.idx3-ubyte'
binfile3 = open(filename3, 'rb')
buf3 = binfile3.read()
index3 = 0
# '>IIII'使用大端法读取四个unsigned int32
magic3, numImages3, numRows3, numColumns3 = struct.unpack_from(
    '>IIII', buf3, index3)
index3 += struct.calcsize('>IIII')

# 读取labels
filename4 = './LAB4/t10k-labels.idx1-ubyte'
binfile4 = open(filename4, 'rb')
buf4 = binfile4.read()

index4 = 0
# '>IIII'使用大端法读取两个unsigned int32
magic4, numLabels4 = struct.unpack_from('>II', buf4, index4)
index4 += struct.calcsize('>II')

test_num = 8
test_mat = zeros((test_num, 28*28))

i = 0
while i < test_num:
    numtemp = struct.unpack_from('1B', buf4, index4)
    label = numtemp[0]
    index4 += struct.calcsize('1B')
    im = struct.unpack_from('>784B', buf3, index3)
    index3 += struct.calcsize('>784B')
    if label == CHOSEN_NUM:
        test_mat[i] = im
        image = array(im).reshape(28, 28)
        plt.subplot(12, 4, i+33)
        plt.imshow(image)
        i += 1

test_normal = test_mat-sum(test_mat, 0)/mat(test_mat).shape[0]
data = dot(dot(test_normal, main_mat.T), main_mat) + \
    sum(test_mat, 0)/mat(test_mat).shape[0]

for i in range(test_num):
    image = array(data[i]).reshape(28, 28).astype(int)
    plt.subplot(12, 4, i+41)
    plt.imshow(image)
plt.show()
