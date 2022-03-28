import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

# TODO: Fix bug
def page_rank(G, s=.85, maxerr=.0001):
    """
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G: matrix representing state transitions
    Gij is a binary value representing a transition from state i to j.
    s: probability of following a transition. 1-s probability of teleporting
    to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will
       have converged.
    """
    n = G.shape[0]
    # 将 G into 马尔科夫 A
    A = coo_matrix(G, dtype=float)
    out_degree = np.array(A.sum(1))[:, 0]
    A.data /= out_degree[A.row]
    sink = out_degree == 0
    # 计算PR值，直到满足收敛条件
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r - ro)) > maxerr:
        ro = r.copy()
        for i in range(0, n):
          Ai = np.array(A[:, i].todense())[:, 0]
          Di = sink / float(n)
          Ei = np.ones(n) / float(n)
          r[i] = ro.dot(Ai * s + Di * s + Ei * (1 - s))
    # 归一化
    return r / float(sum(r))


if __name__ == '__main__':
    # 上面的例子
    G = np.array([[0, 1, 1],
            [0, 0, 1],
            [1, 0, 0]])
    print(page_rank(G, s=0.5))