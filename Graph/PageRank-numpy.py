import numpy as np
from scipy.sparse import csc_matrix, coo_matrix


def page_rank(out_adj, node_values, d=.85, max_err=.0001):
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
    n_nodes = out_adj.shape[0]
    out_degree = np.array(out_adj.sum(1))
    transition_matrix = out_adj.T / out_degree
    # 计算PR值，直到满足收敛条件
    last_node_values = np.zeros(n_nodes)
    n_iter = 0
    while np.sum(np.abs(node_values - last_node_values)) > max_err:
        last_node_values = node_values.copy()
        node_values = (1 - d) / n_nodes + d * (transition_matrix @ node_values)
        n_iter += 1
        print('Iteration: {}, PageRank Value: {}'.format(n_iter, node_values))
    return node_values


if __name__ == '__main__':
    # 上面的例子
    out_adj = np.array([[0, 1, 1],
            [0, 0, 1],
            [1, 0, 0]])
    node_values = np.array([1, 1, 1])
    print(page_rank(out_adj, node_values, d=0.5))