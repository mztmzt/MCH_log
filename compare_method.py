import numpy as np
from scipy.spatial.distance import cdist

def sp_matrix(a):
    row_indices, col_indices = a.nonzero()
    values = a[row_indices, col_indices]
    num = row_indices.shape[0]
    tuple = np.zeros((3, num))
    tuple[0, :] = row_indices + 1
    tuple[1, :] = col_indices + 1
    tuple[2, :] = values
    tuple = tuple.T

    return tuple

def user_nearest_neighbor_fill(X):
    """
    Parameters:
        X: array-like, shape (n_samples, n_features)
    Returns:
        X_new: array-like, shape (n_samples, n_features)
    """
    X_new = X.copy()
    n_samples, n_features = X_new.shape
    for i in range(n_samples):
        for j in range(n_features):
            X_new = X_new.astype('float')
            X_new[X_new == 0] = np.nan
            if np.isnan(X_new[i, j]):
                # Find the average value of the nearest known elements around the missing element
                non_zero_rows = np.where(X[:, j] != 0)[0]
                X_new_2 = X[non_zero_rows, :]
                distances = cdist(X[i, :].reshape(1, -1), X_new_2)
                # distances[:, j] = np.inf
                nearest_neighbors = np.argmin(distances, axis=1)
                nearest_neighbor_values = X_new_2[nearest_neighbors, j]
                X_new[i, j] = np.nanmean(nearest_neighbor_values)
    return X_new

def item_nearest_neighbor_fill(X):
    """
    Implementation of the nearest neighbor completion algorithm
    Parameters:
        X: array-like, shape (n_samples, n_features)
           Matrix to be completed
    Returns:
        X_new: array-like, shape (n_samples, n_features)
               Completed matrix
    """
    X = X.T
    X_new = X.copy()
    n_samples, n_features = X_new.shape
    for i in range(n_samples):
        for j in range(n_features):
            X_new = X_new.astype('float')
            X_new[X_new == 0] = np.nan
            if np.isnan(X_new[i, j]):
                # Find the average value of the nearest known elements around the missing element
                non_zero_rows = np.where(X[:, j] != 0)[0]
                X_new_2 = X[non_zero_rows, :]
                distances = cdist(X[i, :].reshape(1, -1), X_new_2)
                # distances[:, j] = np.inf
                nearest_neighbors = np.argmin(distances, axis=1)
                nearest_neighbor_values = X_new_2[nearest_neighbors, j]
                X_new[i, j] = np.nanmean(nearest_neighbor_values)
    return X_new.T

from scipy.sparse import lil_matrix
from sklearn.utils.extmath import randomized_svd

def trust_svd(M, rank=20, gamma=1e-4, max_iter=50):
    """
    TrustSVD matrix completion algorithm
    :param M: Matrix to be completed, may contain missing values (represented by NaN)
    :param rank: Rank of the singular value decomposition
    :param gamma: Regularization coefficient
    :param max_iter: Maximum number of iterations
    :return: Completed matrix
    """
    # Fill missing values with 0
    M[np.isnan(M)] = 0
    # Initialize P, Q, X, Y matrices
    P = np.random.rand(M.shape[0], rank)
    Q = np.random.rand(M.shape[1], rank)
    X = np.random.rand(M.shape[0], rank)
    Y = np.random.rand(M.shape[1], rank)
    # Construct sparse matrix
    M_sparse = lil_matrix(M)
    # Iterative optimization
    for iter in range(max_iter):
        # Fix Q and Y, optimize P and X
        P = M_sparse.dot(Q) @ np.linalg.inv(Q.T @ Q + gamma * np.eye(rank))
        X = M_sparse.T.dot(P) @ np.linalg.inv(P.T @ P + gamma * np.eye(rank))
        # Fix P and X, optimize Q and Y
        Q = M_sparse.T.dot(P) @ np.linalg.inv(P.T @ P + gamma * np.eye(rank))
        Y = M_sparse.dot(Q) @ np.linalg.inv(Q.T @ Q + gamma * np.eye(rank))
    # Complete matrix
    M_completed = X @ Y.T
    # Replace missing values back with NaN
    M_completed[M == 0] = np.nan
    return M_completed
