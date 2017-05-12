# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import *
from loadData import load_rating_data, spilt_rating_dat


class PMF(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=100, num_batches=10,
                 batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop, 预定义的训练epoch数
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_C = None  # Movie feature vectors
        self.w_I = None  # User feature vectors

        self.err_train = []
        self.err_val = []
        self.data = None
        self.train_data = None
        self.test_data = None
        self.train_rmse = []
        self.test_rmse = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and validation data.  ***********#
    # ***************** train_vec=TrainData, val_vec=TestData*************#
    def fit(self, train_vec, val_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值

        pairs_tr = train_vec.shape[0]  # traindata 中条目数,训练数据的条目数
        pairs_va = val_vec.shape[0]  # testdata中条目数,测试数据的条目数
        
        # 1-p-i, 2-m-c

        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(val_vec[:, 0]))) + 1  # 第0列，user总数， 别误会，用户是从0开始的，训练集和测试集中用户id最大的那个加上1就是整个样本空间中的用户个数。
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(val_vec[:, 1]))) + 1  # 第1列，movie总数，这个同理

        incremental = False
        if ((not incremental) or (self.w_C is None)):
            # initialize
            self.epoch = 0
            #N(\mu, \sigma^2) ``sigma * np.random.randn(...) + mu`` 
            self.w_C = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵,标准差为0.1,均值为0
            self.w_I = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵

            self.w_C_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_I_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:
            self.epoch += 1
            # 每次epoch都shuffle训练集，随机梯度下降（SDG）
            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  # 创建等差array，公差为1
            np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱，将shuffled_order打乱

            # Batch update
            for batch in range(self.num_batches):
                print "epoch %d batch %d" % (self.epoch, batch+1)
                batch_idx = np.mod(np.arange(self.batch_size * batch, self.batch_size * (batch + 1)),
                                   shuffled_order.shape[0])  # 本次迭代要使用的索引下标
                
                # 随机取得的用户ID，1000个
                batch_invID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                # 随机取得的电影ID，1000个， batch_invID和batch_comID项对应的
                batch_comID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function， 这个量就是评分，评分矩阵的一行的和。即用户对电影的总体评分。这个量是估计值，使用用户潜在因子矩阵与电影潜在因子矩阵相乘并求行和。
                pred_out = np.sum(np.multiply(self.w_I[batch_invID, :], self.w_C[batch_comID, :]),
                                  axis=1)  # mean_inv subtracted
                # train_vec的第2列是评分，用刚刚计算的评分估计值减掉训练集中的评分，再加上训练集中的评分的均值，即可得到误差，这里为了计算，直接搞出来：rawErr = 均值-(训练值-估计值)
                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients 计算梯度。rawErr[:, np.newaxis]是将rawErr这个向量变1000行， 1列的矩阵, multiply是对应位置相乘，与array中的*等价
                Ix_C = 2 * np.multiply(rawErr[:, np.newaxis], self.w_I[batch_invID, :]) + self._lambda * self.w_C[batch_comID, :]
                Ix_I = 2 * np.multiply(rawErr[:, np.newaxis], self.w_C[batch_comID, :]) + self._lambda * self.w_I[batch_invID, :]

                dw_C = np.zeros((num_item, self.num_feat))
                dw_I = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_C[batch_comID[i], :] += Ix_C[i, :]
                    dw_I[batch_invID[i], :] += Ix_I[i, :]

                # Update with momentum
                self.w_C_inc = self.momentum * self.w_C_inc + self.epsilon * dw_C / self.batch_size
                self.w_I_inc = self.momentum * self.w_I_inc + self.epsilon * dw_I / self.batch_size
                self.w_C = self.w_C - self.w_C_inc
                self.w_I = self.w_I - self.w_I_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_C[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = LA.norm(rawErr) ** 2 + 0.5 * self._lambda * (LA.norm(self.w_I) ** 2 + LA.norm(self.w_C) ** 2)
                    self.err_train.append(np.sqrt(obj / pairs_tr))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(val_vec[:, 0], dtype='int32'), :],
                                                  self.w_C[np.array(val_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - val_vec[:, 2] + self.mean_inv
                    self.err_val.append(LA.norm(rawErr) / np.sqrt(pairs_va))

                    # Print info
                if batch == self.num_batches - 1:
                    print('Training RMSE: %f, Test RMSE %f' % (self.err_train[-1], self.err_val[-1]))
                    self.train_rmse.append(self.err_train[-1])
                    self.test_rmse.append(self.err_val[-1])
                    # ****************Predict rating of all movies for the given user. ***************#

    def predict(self, invID):
        return np.dot(self.w_C, self.w_I[int(invID), :]) + self.mean_inv  # numpy.dot 点乘

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, model, test_vec, k=10):  # model TrainDataSet, test_vec
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv))[-k:]  # numpy.argsort索引排序

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)


if __name__ == "__main__":
    file_path = "data/ml-100k/u.data"
    pmf = PMF()
    ratings = load_rating_data(file_path)
    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    train, test = spilt_rating_dat(ratings)
    pmf.fit(train, test)
# 
    # Check performance by plotting train and test errors
    plt.plot(range(pmf.maxepoch), pmf.train_rmse, marker='o', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.test_rmse, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()



##np.amax的用法
#>>> import numpy as np
#>>> a = np.array([[1, 2, 3, 4, 5, 6],[2, 3, 4, 5, 6, 7],[2, 4, 9, 2, 1, 2],[8, 2, 3, 4, 5, 1]])
#>>> a
#array([[1, 2, 3, 4, 5, 6],
#       [2, 3, 4, 5, 6, 7],
#       [2, 4, 9, 2, 1, 2],
#       [8, 2, 3, 4, 5, 1]])
#>>> np.amax(a)
#9
#>>> np.amax(a, axis=0)
#array([8, 4, 9, 5, 6, 7])
#>>> np.amax(a, axis=1)
#array([6, 7, 9, 8])


##np.random.randn()使用方法
#Notes
#-----
#For random samples from :math:`N(\mu, \sigma^2)`, use:
#
#``sigma * np.random.randn(...) + mu``
#
#Examples
#--------
#>>> np.random.randn()
#2.1923875335537315 #random
#
#Two-by-four array of samples from N(3, 6.25):
#
#>>> 2.5 * np.random.randn(2, 4) + 3
#varray([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],  #random
#      [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]]) #random

# np.mod的用法
# >>> np.mod([0, 1, 2, 3, 4, 5, 6], 4)
# array([0, 1, 2, 3, 0, 1, 2])

