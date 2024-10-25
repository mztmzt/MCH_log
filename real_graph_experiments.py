import networkx
import numpy as np
from sklearn.cluster import SpectralClustering
from random import random
from random import seed
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_absolute_error as MAE

def local_refinement_new(U, labels_, Adj, c, HG, c_2):
    n,m = U.shape

    rating_estimate_0 = (np.sum(U[labels_==0]==1,axis=0)<np.sum(U[labels_==0]==2,axis=0)) +1 #majority voting rule
    rating_estimate_1 = (np.sum(U[labels_==1]==1,axis=0)<np.sum(U[labels_==1]==2,axis=0)) +1
    rating_estimate_2 = (np.sum(U[labels_ == 2] == 1, axis=0) < np.sum(U[labels_ == 2] == 2, axis=0)) + 1

    # score_0 = c*np.sum(Adj[labels_==0],axis=0) + np.sum(U == rating_estimate_0[np.newaxis,:],axis=1)
    # score_1 = c*np.sum(Adj[labels_==1],axis=0) + np.sum(U == rating_estimate_1[np.newaxis,:],axis=1)

    score_HG_0 = np.zeros(n)
    score_HG_1 = np.zeros(n)
    score_HG_2 = np.zeros(n)
    for i in range(n):
        a = HG[:, HG[i,:]==1]

        b0 = a[labels_==0,:]
        c0= np.sum(b0, axis=0)
        score_HG_0[i] = c_2*np.sum((c0>1))

        b1 = a[labels_ == 1, :]
        c1 = np.sum(b1, axis=0)
        score_HG_1[i] = c_2 * np.sum((c1 > 1))

        b2 = a[labels_ == 2, :]
        c2 = np.sum(b2, axis=0)
        score_HG_2[i] = c_2 * np.sum((c2 > 1))

    score_0 = c*np.sum(Adj[labels_==0],axis=0) + score_HG_0 + np.sum(U == rating_estimate_0[np.newaxis,:],axis=1)
    score_1 = c * np.sum(Adj[labels_ == 1], axis=0) + score_HG_1 + np.sum(U == rating_estimate_1[np.newaxis, :], axis=1)
    score_2 = c * np.sum(Adj[labels_ == 2], axis=0) + score_HG_2 + np.sum(U == rating_estimate_2[np.newaxis, :], axis=1)

    labels_new = np.zeros(len(score_0), dtype=int)
    for i in range(len(score_0)):
        max_value = max(score_0[i], score_1[i], score_2[i])
        if max_value == score_0[i]:
            labels_new[i] = 0
        elif max_value == score_1[i]:
            labels_new[i] = 1
        else:
            labels_new[i] = 2

    # labels_new = ((score_0 < score_1)).astype(float)
    # labels_new = np.squeeze(np.asarray((labels_new)))
    label0 = (labels_new == 0).astype(int)
    label1 = (labels_new == 1).astype(int)
    label2 = (labels_new == 2).astype(int)

    return np.outer(label0, rating_estimate_0)+np.outer(label1, rating_estimate_1)+np.outer(label2, rating_estimate_2), labels_new, rating_estimate_0, rating_estimate_1, rating_estimate_2

def open_file(path):
    with open(path, 'r') as f:
        out = []
        for line in f:
            elements = line.strip().split(',')
            values = [int(e) for e in elements]
            out.append(values)
    return out

def generate_A_from_H(inc_HG):
    d_v = np.sum(inc_HG, axis=1)
    d_v_1 = 1 / np.sqrt(d_v)
    d_e = np.sum(inc_HG, axis=0)
    d_e_1 = 1 / d_e
    d_e_1 = np.diag(d_e_1)
    # adj_HG = inc_HG @ d_e_1
    # adj_HG = inc_HG @ d_e_1
    # d_v = np.sum(adj_HG, axis=1)
    # d_v_1 = 1 / np.sqrt(d_v)
    d_v_1 = np.diag(d_v_1)
    # calculate weighted adj matrix
    adj_HG = inc_HG @ d_e_1 @ inc_HG.T
    # adj_HG = d_v_1 @ inc_HG @ inc_HG.T @ d_v_1
    return adj_HG

from sklearn.cluster import KMeans

def weighted_spectral_clustering(W, k):
    # calculate weighted adj matrix
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    # calculate vector
    eigvals, eigvecs = np.linalg.eig(L)
    indices = np.argsort(eigvals)[:k]
    features = eigvecs[:, indices]

    # normalization
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    labels = kmeans.labels_

    return labels

def local_refinement_new(U, labels_, Adj, c, HG, c_2):
    n,m = U.shape

    rating_estimate_0 = (np.sum(U[labels_ == 0] == 1, axis=0) < np.sum(U[labels_ == 0] == 2, axis=0)) + 1 #majority voting rule
    rating_estimate_1 = (np.sum(U[labels_ == 1] == 1, axis=0) < np.sum(U[labels_ == 1] == 2, axis=0)) + 1
    rating_estimate_2 = (np.sum(U[labels_ == 2] == 1, axis=0) < np.sum(U[labels_ == 2] == 2, axis=0)) + 1
    rating_estimate_3 = (np.sum(U[labels_ == 3] == 1, axis=0) < np.sum(U[labels_ == 3] == 2, axis=0)) + 1
    rating_estimate_4 = (np.sum(U[labels_ == 4] == 1, axis=0) < np.sum(U[labels_ == 4] == 2, axis=0)) + 1
    rating_estimate_5 = (np.sum(U[labels_ == 5] == 1, axis=0) < np.sum(U[labels_ == 5] == 2, axis=0)) + 1
    rating_estimate_6 = (np.sum(U[labels_ == 6] == 1, axis=0) < np.sum(U[labels_ == 6] == 2, axis=0)) + 1
    rating_estimate_7 = (np.sum(U[labels_ == 7] == 1, axis=0) < np.sum(U[labels_ == 7] == 2, axis=0)) + 1
    rating_estimate_8 = (np.sum(U[labels_ == 8] == 1, axis=0) < np.sum(U[labels_ == 8] == 2, axis=0)) + 1

    # score_0 = c*np.sum(Adj[labels_==0],axis=0) + np.sum(U == rating_estimate_0[np.newaxis,:],axis=1)
    # score_1 = c*np.sum(Adj[labels_==1],axis=0) + np.sum(U == rating_estimate_1[np.newaxis,:],axis=1)

    score_HG_0 = np.zeros(n)
    score_HG_1 = np.zeros(n)
    score_HG_2 = np.zeros(n)
    score_HG_3 = np.zeros(n)
    score_HG_4 = np.zeros(n)
    score_HG_5 = np.zeros(n)
    score_HG_6 = np.zeros(n)
    score_HG_7 = np.zeros(n)
    score_HG_8 = np.zeros(n)

    for i in range(n):
        a = HG[:, HG[i,:]==1]

        b0 = a[labels_== 0,:]
        c0= np.sum(b0, axis=0)
        score_HG_0[i] = c_2*np.sum((c0 > 1))

        b1 = a[labels_ == 1, :]
        c1 = np.sum(b1, axis=0)
        score_HG_1[i] = c_2 * np.sum((c1 > 1))

        b2 = a[labels_ == 2, :]
        c2 = np.sum(b2, axis=0)
        score_HG_2[i] = c_2 * np.sum((c2 > 1))

        b3 = a[labels_ == 3, :]
        c3 = np.sum(b3, axis=0)
        score_HG_3[i] = c_2 * np.sum((c3 > 1))

        b4 = a[labels_ == 4, :]
        c4 = np.sum(b4, axis=0)
        score_HG_4[i] = c_2 * np.sum((c4 > 1))

        b5 = a[labels_ == 5, :]
        c5 = np.sum(b5, axis=0)
        score_HG_5[i] = c_2 * np.sum((c5 > 1))

        b6 = a[labels_ == 6, :]
        c6 = np.sum(b6, axis=0)
        score_HG_6[i] = c_2 * np.sum((c6 > 1))

        b7 = a[labels_ == 7, :]
        c7 = np.sum(b7, axis=0)
        score_HG_7[i] = c_2 * np.sum((c7 > 1))

        b8 = a[labels_ == 8, :]
        c8 = np.sum(b8, axis=0)
        score_HG_8[i] = c_2 * np.sum((c8 > 1))



    score_0 = c * np.sum(Adj[labels_ == 0], axis=0) + score_HG_0 + np.sum(U == rating_estimate_0[np.newaxis, :], axis=1)
    score_1 = c * np.sum(Adj[labels_ == 1], axis=0) + score_HG_1 + np.sum(U == rating_estimate_1[np.newaxis, :], axis=1)
    score_2 = c * np.sum(Adj[labels_ == 2], axis=0) + score_HG_2 + np.sum(U == rating_estimate_2[np.newaxis, :], axis=1)
    score_3 = c * np.sum(Adj[labels_ == 3], axis=0) + score_HG_3 + np.sum(U == rating_estimate_3[np.newaxis, :], axis=1)
    score_4 = c * np.sum(Adj[labels_ == 4], axis=0) + score_HG_4 + np.sum(U == rating_estimate_4[np.newaxis, :], axis=1)
    score_5 = c * np.sum(Adj[labels_ == 5], axis=0) + score_HG_5 + np.sum(U == rating_estimate_5[np.newaxis, :], axis=1)
    score_6 = c * np.sum(Adj[labels_ == 6], axis=0) + score_HG_6 + np.sum(U == rating_estimate_6[np.newaxis, :], axis=1)
    score_7 = c * np.sum(Adj[labels_ == 7], axis=0) + score_HG_7 + np.sum(U == rating_estimate_7[np.newaxis, :], axis=1)
    score_8 = c * np.sum(Adj[labels_ == 8], axis=0) + score_HG_8 + np.sum(U == rating_estimate_8[np.newaxis, :], axis=1)

    labels_new = np.zeros(len(score_0), dtype=int)
    for i in range(len(score_0)):
        max_value = max(score_0[i], score_1[i], score_2[i], score_3[i], score_4[i], score_5[i], score_6[i], score_7[i], score_8[i])
        if max_value == score_0[i]:
            labels_new[i] = 0
        elif max_value == score_1[i]:
            labels_new[i] = 1
        elif max_value == score_2[i]:
            labels_new[i] = 2
        elif max_value == score_3[i]:
            labels_new[i] = 3
        elif max_value == score_4[i]:
            labels_new[i] = 4
        elif max_value == score_5[i]:
            labels_new[i] = 5
        elif max_value == score_6[i]:
            labels_new[i] = 6
        elif max_value == score_7[i]:
            labels_new[i] = 7
        else:
            labels_new[i] = 8

    # labels_new = ((score_0 < score_1)).astype(float)
    # labels_new = np.squeeze(np.asarray((labels_new)))
    label0 = (labels_new == 0).astype(int)
    label1 = (labels_new == 1).astype(int)
    label2 = (labels_new == 2).astype(int)
    label3 = (labels_new == 3).astype(int)
    label4 = (labels_new == 4).astype(int)
    label5 = (labels_new == 5).astype(int)
    label6 = (labels_new == 6).astype(int)
    label7 = (labels_new == 7).astype(int)
    label8 = (labels_new == 8).astype(int)

    out = np.outer(label0, rating_estimate_0) + \
          np.outer(label1, rating_estimate_1) + \
          np.outer(label2, rating_estimate_2) + \
          np.outer(label3, rating_estimate_3) + \
          np.outer(label4, rating_estimate_4) + \
          np.outer(label5, rating_estimate_5) + \
          np.outer(label6, rating_estimate_6) + \
          np.outer(label7, rating_estimate_7) + \
          np.outer(label8, rating_estimate_8)

    return out, labels_new

def probabilistic_sampling(a, p):
    mask = np.random.binomial(1, p, size=len(a))
    result = a * mask
    return result

def probabilistic_flip(m, p):
    mask = np.random.binomial(1, p, size=m.shape)
    flipped = np.where(mask, 3 - m, m)
    return flipped


if __name__=='__main__':
    """
    clustering
    """
    num_clusters = 9
    gt_label = np.loadtxt('./dataset/contact-high-school/node-labels-contact-high-school.txt')
    num_users = gt_label.shape[0]
    edge_HG = open_file('./dataset/contact-high-school/hyperedges-contact-high-school.txt')
    num_edges = len(edge_HG)
    inc_HG = np.zeros((num_users, num_edges))
    for i in range(num_edges):
        e_nodes = edge_HG[i]
        for node in e_nodes:
            inc_HG[node-1, i] = 1


    q = 1 # edge sample probability: q

    idx_HG = np.sum(inc_HG, axis=0)
    idx_HG2 = np.where(idx_HG == 2, 1, 0)
    idx_HG2 = probabilistic_sampling(idx_HG2, q)
    inc_HG2 = inc_HG[:, np.where(idx_HG2 == 1)[0]]
    # for i in range(6):
    #     print(len(np.where(np.where(idx_HG == i, 1, 0) == 1)[0]))
    d_v_non = np.sum(inc_HG2, axis=1)
    d_v_non = np.where(d_v_non == 0)
    inc_non = inc_HG[d_v_non[0], :]
    idx_non = np.sum(inc_non, axis=0)
    idx_non = np.where(idx_non > 0, 1, 0)
    idx_HG2_new = idx_HG2 + idx_non
    inc_HG2_new = inc_HG[:, np.where(idx_HG2_new == 1)[0]]

    idx_HG3to5 = np.where(idx_HG > 2, 1, 0)
    idx_HG3to5 = probabilistic_sampling(idx_HG3to5, q)
    inc_HG3to5 = inc_HG[:, np.where(idx_HG3to5 == 1)[0]]
    inc_HG_new = inc_HG[:, np.where(idx_HG2_new + idx_HG3to5 == 1)[0]]

    inc_HG2to5 = inc_HG[:, np.where(idx_HG2 + idx_HG3to5 == 1)[0]]


    # d_v = np.sum(inc_HG, axis=1)
    # d_v = np.where(d_v > 100, 0, d_v)
    # d_v_1 = 1 / np.sqrt(d_v)
    # d_e = np.sum(inc_HG, axis=0)
    # d_e_1 = 1 / d_e
    #
    # d_v_1 = np.diag(d_v_1)
    # d_e_1 = np.diag(d_e_1)
    #
    # # 计算加权邻接矩阵
    # adj_HG = d_v_1 @ inc_HG @ d_e_1  @ inc_HG.T @ d_v_1
    adj_G = generate_A_from_H(inc_HG2_new)
    adj_HG = generate_A_from_H(inc_HG_new)
    # adj_G = inc_HG2 @ inc_HG2.T
    # adj_HG = inc_HG2to5 @ inc_HG2to5.T

    A_G = np.where(adj_G > 0, 1, adj_G)
    A_HG = np.where(adj_HG > 0, 1, adj_HG)

    sc1 = SpectralClustering(n_clusters = num_clusters, affinity='precomputed', n_init=20)
    sc1.fit(adj_G)
    idx1 = sc1.labels_ + 1

    sc1 = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', n_init=20)
    sc1.fit(adj_HG)
    idx2 = sc1.labels_ + 1

    from sklearn.metrics import adjusted_rand_score

    score = adjusted_rand_score(gt_label, idx1)
    # print("G random index:", score)

    score = adjusted_rand_score(gt_label, idx2)
    # print("HG random index:", score)


    """
    Generating matrix
    """
    m = 90
    rating_vetor_1_gt = np.random.randint(1, 3, (1, m))
    rating_vector_mat = np.ones((num_clusters, 1)) @ rating_vetor_1_gt
    for i in range(num_clusters):
        for j in range(10*(i), 10*(i+1)):
            if rating_vetor_1_gt[0, j] == 1:
                rating_vector_mat[i, j] = 2
            else:
                rating_vector_mat[i, j] = 1
    Matrix = np.zeros((num_users, m))
    for i in range(num_users):
        temp = int(gt_label[i] - 1)
        Matrix[i, :] = rating_vector_mat[temp, :]

    p = 0.1
    theta = 0.1
    Matrix_noise = probabilistic_flip(Matrix, theta)
    sample_Matrix = np.random.binomial(n=1, p=p, size=Matrix_noise.shape) * Matrix_noise

    """
    Save Matrix and Trust
    """
    from compare_method import sp_matrix
    I = np.identity(A_G.shape[0])

    GT_sp = sp_matrix(Matrix)
    Sub_sp = sp_matrix(sample_Matrix)
    Adj_sp = sp_matrix(A_G - I)

    fmt = ['%d', '%d', '%.2f']
    np.savetxt('testset.txt', GT_sp, fmt=fmt)
    np.savetxt('trainset.txt', Sub_sp, fmt=fmt)

    np.savetxt('trust.txt', Adj_sp, fmt=fmt)


    """
    My Method
    """
    c_1 = 0.1
    c_2 = 0.1

    numiter = 10
    idx_temp_HG = idx2 - 1
    idx_temp_G = idx1 - 1
    for k1 in range(numiter):
        out_HG, idx_new_HG = local_refinement_new(sample_Matrix, idx_temp_HG, A_G, c_1, inc_HG3to5, c_2)
        inc_HG_zero = np.zeros_like(inc_HG3to5)
        out_G, idx_new_G = local_refinement_new(sample_Matrix, idx_temp_G, A_HG, c_1, inc_HG_zero, 0)

        idx_temp_HG = idx_new_HG
        idx_temp_G = idx_new_G

        score_HG = adjusted_rand_score(gt_label, idx_new_HG)
        score_G = adjusted_rand_score(gt_label, idx_new_G)

        err_mat_HG = MAE(out_HG, Matrix)
        err_mat_G = MAE(out_G, Matrix)

        print("HG:", err_mat_HG, "G:", err_mat_G, 'idx_acc_HG:', score_HG, 'idx_acc_G', score_G)



    from compare_method import user_nearest_neighbor_fill, item_nearest_neighbor_fill
    '''
    User-KNN algorithm
    '''
    Mat_knn_u = user_nearest_neighbor_fill(sample_Matrix)
    err_user_knn = MAE(Mat_knn_u, Matrix)
    print('err_user_knn:', err_user_knn)

    '''
    Item-KNN algorithm
    '''
    Mat_knn_i = item_nearest_neighbor_fill(sample_Matrix)
    err_item_knn = MAE(Mat_knn_i, Matrix)
    print('err_item_knn:', err_item_knn)









