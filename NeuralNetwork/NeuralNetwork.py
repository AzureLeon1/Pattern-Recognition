import numpy as np
import random
import matplotlib.pyplot as plt


def read_data():
    """从文件中读取训练样本"""
    data = np.genfromtxt('../datasets/NNData/data.csv', delimiter=',')
    features = data[:, :3]
    labels = data[:, 3]
    labels = np.array(list(map(int, labels)))
    return features, labels


def sigmoid(x):
    """激活函数sigmoid"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """sigmoid求导"""
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    """激活函数tanh"""
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(x):
    """tanh求导"""
    return 1 - tanh(x) ** 2


def rand(a, b):
    return (b - a) * random.random() + a


class MLP:
    def __init__(self, dim_input, dim_hidden, dim_output):
        # 网络结构
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        # w_i_h:输入层到隐藏层的初始连接权 dim_hidden * dim_input 维，对应每个输入结点到隐藏结点的连接权
        self.w_i_h = self.get_w(dim_hidden, dim_input)
        # w_h_o:隐藏层到输出层的初始连接权 dim_output * dim_hidden 维，对应每个隐藏结点到输出结点的连接权
        self.w_h_o = self.get_w(dim_output, dim_hidden)
        # 输入层
        self.input = np.zeros([dim_input, 1])
        # 隐藏层输入
        self.net_h = np.zeros([dim_hidden, 1])
        # 隐藏层输出
        self.y = np.zeros([dim_hidden, 1])
        # 输出层输入
        self.net_o = np.zeros([dim_output, 1])
        # 输出层输出
        self.z = np.zeros([dim_output, 1])

    def get_w(self, m, n):
        """初始化连接权重"""
        w = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                w[i, j] = rand(-1, 1)
        return w

    def forward(self, feature):
        """前向传播"""
        self.input = feature.reshape(-1, 1)
        self.net_h = self.w_i_h.dot(self.input)
        self.y = tanh(self.net_h)
        self.net_o = self.w_h_o.dot(self.y)
        self.z = sigmoid(self.net_o)


    def backward(self, label, learning_rate):
        """反向传播"""
        # 得到误差信号
        t = np.zeros(self.dim_output).reshape(-1, 1)
        # print(type(label), label)
        t[label, 0] = 1
        delta_z = (t - self.z) * sigmoid_derivative(self.net_o)  # c * 1维 对应元素相乘，每一个元素代表该输出结点收集到的信号
        delta_h = (self.w_h_o.T).dot(delta_z) * tanh_derivative(self.net_h)  # dim_hidden * 1维 每一个元素代表该隐层结点收集到的信号
        # 更新隐层到输出层权重
        d2 = learning_rate * delta_z.dot(self.y.T)  # c * dim_hidden 每一行代表到该输出结点的连接权更新值
        # 更新输入层到隐层权重
        d1 = learning_rate * delta_h.dot(self.input.T)  # dim_hidden * d 每一行代表到该隐层结点的连接权更新值
        return d1, d2  # 返回该样本对权重的更新


def single_sample_bp(features, labels, network: MLP, max_step=100000, learning_rate=0.1, theta=1e-5):
    """单样本方式更新权重"""
    d1 = np.zeros_like(network.w_i_h)
    d2 = np.zeros_like(network.w_h_o)
    steps = []
    err = []
    best_step = 0
    max_acc = 0
    cnt_convergence = 0
    for step in range(0, max_step):
        k = random.randint(0, len(features) - 1)
        network.forward(features[k])  # 前向传播
        d1, d2 = network.backward(labels[k], learning_rate)  # 反向传播
        network.w_i_h += d1  # 更新权重
        network.w_h_o += d2
        if step % 100 == 0:
            error = test(features, labels, network)
            acc = evaluate(features, labels, network)
            if acc <= max_acc:
                cnt_convergence += 1
            else:
                max_acc = acc
                best_step = step
                cnt_convergence = 0
            # if cnt_convergence >= 5:
            if acc == 1.0:
                steps.append(best_step)
                err.append(error)
                break
            err.append(error)
            steps.append(step)
    return network, err, steps


def batch_bp(features, labels, network: MLP, max_batch=3333, learning_rate=0.1, theta=1e-5):
    """批量更新算法"""
    steps = []
    err = []
    best_step = 0
    max_acc = 0
    cnt_convergence = 0

    dd1 = np.zeros_like(network.w_i_h)
    dd2 = np.zeros_like(network.w_h_o)

    num_samples = len(features)

    for batch in range(max_batch):
        for k in range(num_samples):
            network.forward(features[k])  # 前向传播
            d1, d2 = network.backward(labels[k], learning_rate)  # 反向传播
            dd1 += d1  # 存储权重
            dd2 += d2

        network.w_i_h += dd1  # 更新权重
        network.w_h_o += dd2

        dd1 = np.zeros_like(network.w_i_h)
        dd2 = np.zeros_like(network.w_h_o)
        if batch % 33 == 0:
            error = test(features, labels, network)
            acc = evaluate(features, labels, network)
            if acc <= max_acc:
                cnt_convergence += 1
            else:
                max_acc = acc
                best_step = batch
                cnt_convergence = 0
            # if cnt_convergence >= 4:
            if acc == 1.0:
                steps.append(best_step)
                err.append(error)
                break
            err.append(error)
            steps.append(batch)
    return network, err, steps


def test(features, labels, network: MLP):
    """测试函数，返回square error"""
    error = 0
    num_samples = len(features)
    for i in range(num_samples):
        network.forward(features[i])
        t = np.zeros(network.dim_output).reshape(-1, 1)
        t[labels[i], 0] = 1
        error += np.sum(np.square(t - network.z))  # 误差（目标）函数的导数
    error = error / (2 * num_samples)
    return error


def evaluate(features, labels, network: MLP):
    """评估函数，返回分类Accuracy"""
    cnt_positive = 0
    cnt_all = 0
    num_samples = len(features)
    for i in range(num_samples):
        network.forward(features[i])
        prediction = np.argmax(network.z)
        label = labels[i]
        if prediction == label:
            cnt_positive += 1
        cnt_all += 1
    # print('Accuracy: {:.2%}'.format(cnt_positive / cnt_all))
    return cnt_positive/cnt_all


if __name__ == "__main__":
    features, labels = read_data()
    assert len(features) == len(labels)

    lr = 0.1
    hidden = 64
    add_bias = True


    if add_bias:
        features = np.insert(features, 0, np.ones(features.shape[0]), axis=1)
        network = MLP(4, hidden, 3)
        network2 = MLP(4, hidden, 3)
    else:
        network = MLP(3, hidden, 3)
        network2 = MLP(3, hidden, 3)


    network, y1, steps = single_sample_bp(features, labels, network, learning_rate=lr)
    acc = evaluate(features, labels, network)
    print('Accuracy (Single Sample BP): {:.2%}'.format(acc))
    print('Square Error (Single Sample BP): {}'.format(y1[-1]))
    print('Convergence Steps (Single Sample BP): {}'.format(steps[-1]))


    network2, y2, steps2 = batch_bp(features, labels, network2, learning_rate=lr)
    acc2 = evaluate(features, labels, network2)
    print('Accuracy (Batch BP): {:.2%}'.format(acc2))
    print('Square Error (Batch BP): {}'.format(y2[-1]))
    print('Convergence Steps (Batch BP): {}'.format(steps2[-1]))


    plt.plot(steps[:-1], y1[:-1], 'r')
    plt.xlabel('Iteration Steps (Single Sample BP)')
    plt.ylabel('Loss Function Value')
    plt.title('single-bp')
    plt.savefig("./img/single-bp.png", dpi=300)
    plt.show()

    plt.plot(steps2[:-1], y2[:-1])
    plt.xlabel('Iteration Steps (Batch BP)')
    plt.ylabel('Loss Function Value')
    plt.title('batch-bp')
    plt.savefig("./img/batch-bp.png", dpi=300)
    plt.show()




