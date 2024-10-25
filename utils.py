import math
import numpy as np
from random import random


def generate_SBM(n, p_in, p_out, communities):
    """
    Generates a 3-uniform hypergraph SBM with n nodes and k communities.

    Arguments:
    n -- the number of nodes in the network
    k -- the number of communities or groups
    p_in -- the probability of an edge between three nodes within the same community
    p_out -- the probability of an edge between three nodes in different communities
    communities -- a 1D numpy array representing the ground truth community assignment of the nodes

    Returns:
    adjacency_matrix -- a 3D numpy array representing the adjacency matrix of the hypergraph
    incidence_matrix -- a 2D numpy array representing the incidence matrix of the hypergraph
    """

    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((n, n))
    adjacency_matrix_2 = np.zeros((n, n))

    # Populate the adjacency matrix
    for i in range(n):
        for j in range(i+1, n):
            if communities[i] == communities[j]:
                if np.random.rand() < p_in:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix_2[i, j] = 1
                    adjacency_matrix_2[j, i] = 1  # for spectral clustering using
            else:
                if np.random.rand() < p_out:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix_2[i, j] = 1
                    adjacency_matrix_2[j, i] = 1  # for spectral clustering using

    he = np.where(adjacency_matrix == 1)
    num_he = he[0].shape[0]
    incidence_matrix = np.zeros((n, num_he))
    for i in range(num_he):
        incidence_matrix[he[0][i], i] = 1
        incidence_matrix[he[1][i], i] = 1


    return adjacency_matrix_2, incidence_matrix

def Generate_3_uni_HSBM(numV, p_intra, p_inter, IDX, alloc=200):
    '''

    :param numV: number of nodes
    :param p_intra: in_cluster HG prob
    :param p_inter: cross_cluster HG prob
    :param IDX: label
    :return: HG set
    '''
    offset = 0
    edges = np.zeros((3, alloc))

    for id1 in range(numV):
        for id2 in range(id1 + 1, numV):
            for id3 in range(id2 + 1, numV):
                if IDX[id1] == IDX[id2] and IDX[id2] == IDX[id3]:
                    value = random()
                    if value < p_intra:
                        if offset >= alloc:
                            edges = np.append(edges, np.zeros((3, alloc)), 1)
                            alloc = alloc * 2
                        edges[0, offset] = id1
                        edges[1, offset] = id2
                        edges[2, offset] = id3
                        offset = offset + 1
                else:
                    value = random()
                    if value < p_inter:
                        if offset >= alloc:
                            edges = np.append(edges, np.zeros((3, alloc)), 1)
                            alloc = alloc * 2
                        edges[0, offset] = id1
                        edges[1, offset] = id2
                        edges[2, offset] = id3
                        offset = offset + 1
    edges = edges[:, 1: offset]
    I = np.zeros((numV, edges.shape[1]))
    for i in range(edges.shape[1]):
        I[int(edges[0, i]), i] = 1
        I[int(edges[1, i]), i] = 1
        I[int(edges[2, i]), i] = 1
    return I

def generate_3_uniform_hypergraph_SBM(n, p_in, p_out, communities):
    """
    Generates a 3-uniform hypergraph SBM with n nodes and k communities.

    Arguments:
    n -- the number of nodes in the network
    k -- the number of communities or groups
    p_in -- the probability of an edge between three nodes within the same community
    p_out -- the probability of an edge between three nodes in different communities
    communities -- a 1D numpy array representing the ground truth community assignment of the nodes

    Returns:
    adjacency_matrix -- a 3D numpy array representing the adjacency matrix of the hypergraph
    incidence_matrix -- a 2D numpy array representing the incidence matrix of the hypergraph
    """

    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((n, n, n))

    # Populate the adjacency matrix
    for i in range(n):
        for j in range(i+1, n):
            for l in range(j+1, n):
                if communities[i] == communities[j] == communities[l]:
                    if np.random.rand() < p_in:
                        adjacency_matrix[i, j, l] = 1
                else:
                    if np.random.rand() < p_out:
                        adjacency_matrix[i, j, l] = 1

    he = np.where(adjacency_matrix == 1)
    num_he = he[0].shape[0]
    incidence_matrix = np.zeros((n, num_he))
    for i in range(num_he):
        incidence_matrix[he[0][i], i] = 1
        incidence_matrix[he[1][i], i] = 1
        incidence_matrix[he[2][i], i] = 1

    return adjacency_matrix, incidence_matrix

def comb_compute(n, m):
    return math.factorial(n)//(math.factorial(m)*math.factorial(n-m))

def GroundTruth(Ground_Rating, n, m, k, k2):
    submatrices1=[]
    for i in range(k):
        submatrices2=[]
        for j in range(k2):
            submatrices2.append(Ground_Rating[i][j]*np.ones((int(n/k),int(m/k2))))
        submatrices1.append(np.hstack(submatrices2))
    return np.vstack(submatrices1)

def subMatrix(rating,p, q, theta, n, m, k, k2):
    P = p*theta*np.ones(q+1) #considering noise for selected ones
    P[0]=1-p # not selected ones
    P[int(rating)]=p*(1-(q-1)*theta)
    out = np.random.choice(q+1,(int(n/k),int(m/k2)),p=P)
    return np.random.choice(q+1,(int(n/k),int(m/k2)),p=P)

def MatrixGeneration(p, Ground_Rating, k, k2, theta, q, n, m):
    submatrices1=[]
    for i in range(k):
        submatrices2=[]
        for j in range(k2):
            submatrices2.append(subMatrix(Ground_Rating[i][j],p, q, theta, n, m, k, k2))
        submatrices1.append(np.hstack(submatrices2))
    return np.vstack(submatrices1)