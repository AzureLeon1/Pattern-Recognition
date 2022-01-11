# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def load_data1():
    data_path = '../datasets/ClusteringData/data1.mat'
    data = loadmat(data_path)
    data = data['X']
    data = np.insert(data, 2, 0, axis=1)  # 新增一列，用于标记每个样本的真实类别
    for i in range(5):
        data[i*200:(i+1)*200, 2] = i
    return data


class KMeans:

    def __init__(self, data, k=5, random=1):
        """
        给定数据，初始化类中心
        :param data: 数据
        :param k: 簇的个数
        :param random: 初始化类中心的策略
        """
        self.k = k
        self.nodes = data
        self.nodes = np.insert(self.nodes, 3, 0, axis=1) # 新增一列，用于标记每个样本的预测类别
        self.centers = np.zeros((k, 3))  # 类中心
        if random == 1:  # 随机在样本点中选择初始类中心
            np.random.shuffle(self.nodes)
            for i in range(0, k):
                self.centers[i][0] = self.nodes[i][0]
                self.centers[i][1] = self.nodes[i][1]
                self.centers[i][2] = i
        elif random == 2:  # 分别在每一类中选一点作为初始中心
            for i in range(0, k):
                self.centers[i][0] = self.nodes[i * 200][0]
                self.centers[i][1] = self.nodes[i * 200][1]
                self.centers[i][2] = i
        elif random == 3:  # 选取前n个点作为中心点，都属于第一类
            for i in range(0, k):
                self.centers[i][0] = self.nodes[i][0]
                self.centers[i][1] = self.nodes[i][1]
                self.centers[i][2] = i
        elif random == 4:  # 中心点在第一类中选择一个，然后选择附近点
            self.centers[0][0] = self.nodes[0][0]
            self.centers[0][1] = self.nodes[0][1]
            self.centers[0][2] = 0
            for i in range(1, k):
                self.centers[i][0] = self.centers[i - 1][0] + 0.1
                self.centers[i][1] = self.centers[i - 1][1] + 0.1
                self.centers[i][2] = i

    def iterate(self):
        """
        K-Means聚类迭代过程
        :return: 聚类结果
        """
        flag = True
        cnt = 0
        n_samples = self.nodes.shape[0]
        n_clusters = self.k
        while (flag):
            flag = False
            cnt += 1
            temp_center = np.zeros((n_clusters, 3))

            # 样本分配到各个簇
            for i in range(n_samples):
                distance = np.zeros(n_clusters)
                for j in range(n_clusters):
                    distance[j] = np.linalg.norm(self.nodes[i][:2] - self.centers[j][:2])
                kk = np.argmin(distance)  # 分到该类
                self.nodes[i][3] = kk
                temp_center[kk][0] += self.nodes[i][0]
                temp_center[kk][1] += self.nodes[i][1]
                temp_center[kk][2] += 1

            # 更新类中心点
            for i in range(n_clusters):
                temp_x = temp_center[i][0] / temp_center[i][2]
                temp_y = temp_center[i][1] / temp_center[i][2]
                # 收敛条件
                if temp_x != self.centers[i][0] or temp_y != self.centers[i][1]:
                    flag = True
                self.centers[i][0] = temp_x
                self.centers[i][1] = temp_y
        print("模型经过 %d 轮迭代后收敛" % cnt)
        # 将样本点按照分类划分
        self.nodes_ps = []
        for i in range(0, n_clusters):
            self.nodes_ps.append([])
        for i in range(0, n_samples):
            self.nodes_ps[int(self.nodes[i][3])].append(self.nodes[i])

    def evaluate(self, ground_truth):
        """第一步计算聚类准确率Accuracy，第二步计算类中心mse"""
        n_clusters = self.k
        n_samples = self.nodes.shape[0]

        # 预测类中心与真实类中心对齐
        flag = np.zeros(n_clusters)  # 标记learned centers中的点是否已选
        for i in range(n_clusters):
            dis_min = 10000
            temp = -1
            for j in range(n_clusters):
                dis = pow(ground_truth[i][0] - self.centers[j][0], 2) + pow(ground_truth[i][1] - self.centers[j][1], 2)
                if dis < dis_min and flag[j] == 0:
                    temp = j
                    dis_min = dis
            self.centers[temp][2] = i
            flag[temp] = 1
        old_centers_order = list(range(5))
        new_centers_order = list(self.centers[:,2])
        self.centers = np.array(sorted(self.centers, key=lambda x: x[2]))
        dictionary = dict(zip(old_centers_order, new_centers_order))
        origin_prediction = self.nodes[:,3]
        new_prediction = np.array(list(map(dictionary.get, origin_prediction)))
        self.nodes[:,3] = new_prediction

        # 计算聚类准确率
        n_true_positive_samples = (self.nodes[:, 2] == self.nodes[:, 3]).sum()
        accuracy = n_true_positive_samples / n_samples
        print('聚类准确率：{:.2%}'.format(accuracy))

        # 计算类中心误差
        mse = 0
        for i in range(n_clusters):
            mse += pow(self.centers[i][0] - ground_truth[i][0], 2) + pow(self.centers[i][1] - ground_truth[i][1], 2)
            print("Learned Center %d : %f  %f" % (i, self.centers[i][0], self.centers[i][1]))
            print("Ground Truth %d : %f  %f" % (i, ground_truth[i][0], ground_truth[i][1]))
        print("聚类中心误差（MSE）: %f" % mse)

    def vis_result(self, random_choice):
        """聚类结果可视化"""
        colors = ['r', 'b', 'k', 'g', 'm']
        for i in range(0, self.k):
            nodes = np.array(self.nodes_ps[i])
            plt.scatter(nodes[:, 0], nodes[:, 1], color=colors[i])
        self.centers = np.array(self.centers)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], color='y', marker='+')
        plt.savefig("./img/p1-random"+str(random_choice)+".png", dpi=300)
        plt.show()


if __name__ == "__main__":
    data = load_data1()
    random_choice = 4
    model = KMeans(data=data, random=random_choice)
    model.iterate()
    ground_truth = np.array([[1, -1], [5.5, -4.5], [1, 4], [6, 4.5], [9, 0]])
    model.evaluate(ground_truth)
    model.vis_result(random_choice)
