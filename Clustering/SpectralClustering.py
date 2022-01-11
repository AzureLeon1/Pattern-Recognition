# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import math


def load_data2():
    data_path = '../datasets/ClusteringData/data2.mat'
    data = loadmat(data_path)
    data = data['X']
    data = np.insert(data, 2, 0, axis=1)  # 新增一列，用于标记每个样本的真实类别
    for i in range(2):
        data[i*100:(i+1)*100, 2] = i
    return data


def build_knn_graph(W, k=10):
    idx = np.argsort(W, axis=1)[:, -k:]
    W2 = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(k):
            W2[i][idx[i][j]] = W[i][idx[i][j]]
    return W2


class SpectralClustering:
    def __init__(self, knn_k=10, sigma=2.50, n_clusters=2, sim_fn='rbf', norm='rw'):
        self.knn_k = knn_k
        self.sigma = sigma
        self.n_clusters = n_clusters
        self.norm = norm
        self.sim_fn = sim_fn

    # 高斯径向基核函数
    def rbf(self, x, y):
        return np.exp(-1.0 * (x - y).T @ (x - y) / (2 * self.sigma ** 2))

    # 欧式距离
    def euclid(self, x1, x2, sqrt_flag=False):
        res = np.sum((x1-x2)**2)
        if sqrt_flag:
            res = np.sqrt(res)
        return res


    def fit_predict(self, data):
        # 按照相似度函数计算样本之间的距离
        metric = self.rbf
        if self.sim_fn == 'euclid':
            metric = self.euclid
        W = pairwise_distances(data[:,:2], metric=metric)
        # 对角线置为0
        row, col = np.diag_indices_from(W)
        W[row, col] = 0

        # knn局部图
        W = build_knn_graph(W, self.knn_k)
        W = (W.T + W) / 2

        # 计算度矩阵
        D = np.diag(W.sum(axis=1))

        # 计算拉普拉斯矩阵
        L = D - W

        # 拉普拉斯矩阵规范化
        if self.norm == 'rw':
            L_norm = np.sqrt(np.linalg.inv(D)) @ L @ np.sqrt(np.linalg.inv(D))
        elif self.norm == 'sys':
            L_norm = np.linalg.inv(D) @ L
        else:
            L_norm = L

        # 特征分解，np.linalg.eig()默认按照特征值升序排序了。
        eigenvals, eigvector = np.linalg.eig(L_norm)

        # 如果没有升序排序，可以这样做
        # 将特征值按照升序排列
        ind = np.argsort(eigenvals)
        eig_vec_sorted = eigvector[:, ind]  # 对应的特征向量也要相应调整顺序

        # 统计特征值中等于0的个数
        n_zero_eigenvals = 0
        for eigenval in eigenvals:
            if math.fabs(eigenval) < 1e-6:
                n_zero_eigenvals += 1
            else:
                break
        # 取出前k个最小的非零特征值对应的特征向量，注意这里的k和要聚类的簇数一样
        Q = eig_vec_sorted[:, n_zero_eigenvals: n_zero_eigenvals + self.n_clusters]


        # 对新构成的Q矩阵进行聚类
        km = KMeans(n_clusters=self.n_clusters)
        Q_abs = np.real(Q)

        # 对Q_abs的行向量聚类，并计算出每个样本所属的类
        y_pred = km.fit_predict(Q_abs)

        # 计算聚类准确率
        n_true_positive_samples = (y_pred == data[:, 2]).sum()
        n_samples = data.shape[0]
        n_true_positive_samples = max(n_true_positive_samples, n_samples-n_true_positive_samples)
        accuracy = n_true_positive_samples / n_samples
        print('聚类准确率：{:.2%}'.format(accuracy))

        # 根据预测的标签画出所有的样本
        plt.scatter(data[:, 0], data[:, 1], c=y_pred)
        plt.savefig("./img/p2-spectral.png", dpi=300)
        plt.show()
        return accuracy


if __name__ == "__main__":
    data = load_data2()

    # # grid search
    # sigmas = np.arange(0.5, 5.5, 0.5)
    # ks = np.arange(5, 65, 5)
    # n_sigmas = len(sigmas)
    # n_ks = len(ks)
    # acc = np.zeros([n_sigmas, n_ks])
    # for i in range(n_sigmas):
    #     for j in range(n_ks):
    #         print('sigma:', sigmas[i], 'k:', ks[j])
    #         model = SpectralClustering(knn_k=ks[j], sigma=sigmas[i], n_clusters=2, norm='rw')
    #         cur_acc = model.fit_predict(data)
    #         acc[i][j] = cur_acc
    # np.save('accuracy.npy', acc)

    model = SpectralClustering(knn_k=10, sigma=1.50, n_clusters=2, norm='rw')
    model.fit_predict(data)