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
    return dot(X, transpose(chosen_pair))


def generate_data(feature, sample, mean, variance):
    X = []
    for f in range(feature):
        X.extend(rd.normal(mean[f], variance[f], sample))
    X = mat(X).reshape(feature, sample).T
    return X


# data property
# FEATURE = 3
# SAMPLE = 1000
# MEAN = [100, 10, 70]
# VARIANCE = [100, 20, 0.1]
# X = generate_data(FEATURE, SAMPLE, MEAN, VARIANCE)
# ax = plt.subplot(1, 2, 1, projection='3d')  # create a 3d project
# ax.scatter(X[:, 0].tolist(), X[:, 1].tolist(), X[:, 2].tolist())
# result = PCA(X, 2)
# plt.subplot(1, 2, 2)
# plt.axis([result.min(), result.max(), result.min(), result.max()])
# plt.scatter(result[:, 0].tolist(), result[:, 1].tolist())
# plt.show()

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
K = 100
# 初始化traingMat
train_mat = zeros((trainingNumbers, 28*28))
# 初始化标签
trainingLabels = []


# 获取经过PCA  处理过的traingMat 和 label
for i in range(trainingNumbers):
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')
    train_mat[i] = im
    # image = array(im).reshape(28, 28)
    # plt.subplot(4, 4, i+1)
    # plt.imshow(image)
    # 读取标签
    numtemp = struct.unpack_from('1B', buf1, index1)
    label = numtemp[0]
    index1 += struct.calcsize('1B')
    trainingLabels.append(label)


PCA_mat = PCA(train_mat.T, K).T

for i in range(trainingNumbers):
    im = PCA_mat[i]
    image = array(im).reshape(int(sqrt(K)), int(sqrt(K))).astype(int)
    plt.subplot(4, 4, i+1)

    plt.imshow(image)
plt.show()
