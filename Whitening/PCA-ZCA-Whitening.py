import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
mu = [0, 0]
sigma = [[5, 4], [4, 5]]  # must be positive semi-definite
n = 1000
x = np.random.multivariate_normal(mu, sigma, size=n).T

# x.shape: (2, 1000)

# 这两个时间序列存储在shape为(2,1000)的NumPy数组 [公式] 中，同时，为了便于稍后可视化，这里固定20个最极端的值，并将它们的索引表示为set1（其余数据点的索引存储在set2中）：
set1 = np.argsort(np.linalg.norm(x, axis=0))[-40:]
set2 = np.array(list(set(range(n)) - set(set1)))

evals, evecs = np.linalg.eigh(sigma)

plt.rcParams['figure.figsize'] = (5.0, 5.0) # 设置figure_size尺寸
x1 = x.T[set1].T
x2 = x.T[set2].T
fig = plt.figure()
ax = plt.subplot()
ax.scatter(x1[0,:], x1[1,:], c='tomato', alpha=0.5)
ax.scatter(x2[0,:], x2[1,:], c='mediumturquoise', alpha=0.6)  # 改变颜色

ax_lim = np.absolute(x).max() + 1
ax.set_xlim([-ax_lim, ax_lim])
ax.set_ylim([-ax_lim, ax_lim])

plt.savefig('./img/origin.png')
plt.show()


# PCA 白化
z = np.diag(evals**(-1/2)) @ evecs.T @ x

z1 = z.T[set1].T
z2 = z.T[set2].T
fig = plt.figure()
ax = plt.subplot()
ax.scatter(z1[0,:], z1[1,:], c='tomato', alpha=0.5)
ax.scatter(z2[0,:], z2[1,:], c='mediumturquoise', alpha=0.6)  # 改变颜色

# ax_lim = np.absolute(z).max() + 1
ax.set_xlim([-ax_lim, ax_lim])
ax.set_ylim([-ax_lim, ax_lim])

plt.savefig('./img/PCA.png')
plt.show()

# ZCA 白化

z = evecs @ np.diag(evals**(-1/2)) @ evecs.T @ x

z1 = z.T[set1].T
z2 = z.T[set2].T
fig = plt.figure()
ax = plt.subplot()
ax.scatter(z1[0,:], z1[1,:], c='tomato', alpha=0.5)
ax.scatter(z2[0,:], z2[1,:], c='mediumturquoise', alpha=0.6)  # 改变颜色

# ax_lim = np.absolute(z).max() + 1
ax.set_xlim([-ax_lim, ax_lim])
ax.set_ylim([-ax_lim, ax_lim])

plt.savefig('./img/ZCA.png')
plt.show()