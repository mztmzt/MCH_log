import networkx
import numpy as np
from sklearn.cluster import SpectralClustering
from utils import comb_compute, GroundTruth, MatrixGeneration, generate_SBM, Generate_3_uni_HSBM
from random import random
import math
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')


class SBM(object):

    def __init__(self, num_vertices, alpha, beta, k):
        self.num_vertices = num_vertices
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.G = self.generate(self.num_vertices, self.alpha, self.beta,k)
        self.block_matrix = networkx.adjacency_matrix(self.G)

    def generate(self, num_vertices, alpha, beta, k):

        cluster_num_vertices=int(num_vertices/k)

        sizes = (cluster_num_vertices*np.ones(k)).astype(int).tolist()

        probs = beta * np.ones((k,k))
        for i in range(k):
            probs[i][i]=alpha

        probs = probs.tolist()
        g = networkx.stochastic_block_model(sizes,probs,seed=0)

        return g

def c_estimate_user(idx1_0, Adj):

    n_M = sum(idx1_0==1)
    n_W = sum(idx1_0==0)

    e_b_MW0 = np.linalg.multi_dot([(idx1_0==1),Adj,(idx1_0==0)])
    e_b_MM0 = (np.linalg.multi_dot([(idx1_0==1),Adj,(idx1_0==1)])-n_M)/2
    e_b_WW0 = (np.linalg.multi_dot([(idx1_0==0),Adj,(idx1_0==0)])-n_W)/2

    alpha1_e = (e_b_MM0+e_b_WW0)/((n_M*(n_M-1)/2)+(n_W*(n_W-1)/2))
    beta1_e = (e_b_MW0)/n_M/n_W

    return np.log(alpha1_e*(1-beta1_e)/beta1_e/(1-alpha1_e))/np.log((1-theta)/theta)

def c_estimate_G(idx, G, n, theta):

    # compute_num_edges
    C_1 = G[idx == 0, :]
    C_2 = G[idx == 1, :]
    C_3 = G[idx == 2, :]
    C_1 = np.sum(C_1, axis=0)
    C_2 = np.sum(C_2, axis=0)
    C_3 = np.sum(C_3, axis=0)
    # num_intra = np.sum((M == 3)) + np.sum(W == 3)
    # num_inter = HG.shape[1] - num_intra

    h_C11 = np.sum((C_1==2))
    h_C22 = np.sum((C_2==2))
    h_C33 = np.sum((C_3==2))
    h_C123 = G.shape[1] - h_C11 -h_C22 - h_C33

    alpha2_e = (h_C11 + h_C22) / (3 * comb_compute(int(n/3), 2))
    beta2_e = h_C123 / (comb_compute(n, 2) - 3*comb_compute(int(n/3), 2))
    # print('alpha2_g', alpha2_e)
    # print('beta2_g', beta2_e)

    c_G_e = np.log(alpha2_e*(1-beta2_e)/beta2_e/(1-alpha2_e))/np.log((1-theta)/theta)

    return c_G_e

def c_estimate_HG(idx, HG, n, theta):

    # compute_num_edges
    C_1 = HG[idx == 0, :]
    C_2 = HG[idx == 1, :]
    C_3 = HG[idx == 2, :]
    C_1 = np.sum(C_1, axis=0)
    C_2 = np.sum(C_2, axis=0)
    C_3 = np.sum(C_3, axis=0)
    # num_intra = np.sum((M == 3)) + np.sum(W == 3)
    # num_inter = HG.shape[1] - num_intra

    h_C11 = np.sum((C_1==3))
    h_C22 = np.sum((C_2==3))
    h_C33 = np.sum((C_3==3))
    h_C123 = HG.shape[1] - h_C11 -h_C22 - h_C33

    alpha2_e = (h_C11 + h_C22) / (3 * comb_compute(int(n/3), 3))
    beta2_e = h_C123 / (comb_compute(n, 3) - 3*comb_compute(int(n/3), 3))
    # print('alpha2_hg', alpha2_e)
    # print('beta2_hg', beta2_e)

    c_HG_e = np.log(alpha2_e*(1-beta2_e)/beta2_e/(1-alpha2_e))/np.log((1-theta)/theta)

    return c_HG_e


def local_refinement(M,labels_,Adj,c):
    rating_estimate_0 = (np.sum(M[labels_==0]==1,axis=0)<np.sum(M[labels_==0]==2,axis=0)) +1 #majority voting rule
    rating_estimate_1 = (np.sum(M[labels_==1]==1,axis=0)<np.sum(M[labels_==1]==2,axis=0)) +1
    rating_estimate_2 = (np.sum(M[labels_ == 2] == 1, axis=0) < np.sum(M[labels_ == 2] == 2, axis=0)) + 1

    score_0 = c*np.sum(Adj[labels_==0],axis=0) + np.sum(M == rating_estimate_0[np.newaxis,:],axis=1)
    score_1 = c*np.sum(Adj[labels_==1],axis=0) + np.sum(M == rating_estimate_1[np.newaxis,:],axis=1)
    score_2 = c*np.sum(Adj[labels_==2],axis=0) + np.sum(M == rating_estimate_2[np.newaxis,:],axis=1)

    labels_new = np.zeros(len(score_0), dtype=int)  # Store the category corresponding to each position
    for i in range(len(score_0)):
        # Find the maximum value at each position
        max_value = max(score_0[i], score_1[i], score_2[i])
        # Store the category corresponding to the maximum value in labels
        if max_value == score_0[i]:
            labels_new[i] = 0
        elif max_value == score_1[i]:
            labels_new[i] = 1
        else:
            labels_new[i] = 2

    # labels_new = ((score_0 < score_1)).astype(float)
    # labels_new = np.squeeze(np.asarray((labels_new)))
    label0 = (labels_new==0).astype(int)
    label1 = (labels_new==1).astype(int)
    label2 = (labels_new == 2).astype(int)

    return np.outer(label0, rating_estimate_0)+np.outer(label1, rating_estimate_1)+np.outer(label2, rating_estimate_2), labels_new, rating_estimate_0, rating_estimate_1, rating_estimate_2


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

    labels_new = np.zeros(len(score_0), dtype=int)  # Store the category corresponding to each position
    for i in range(len(score_0)):
        # find max value
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


if __name__=='__main__':
    from sklearn.metrics import mean_absolute_error as MAE

    #parameter setting
    n = 300
    m = 100
    gamma = 0.4
    theta = 0.1

    k = 3
    k2 = m
    q = 2  # rating levels

    #SBM parameter
    I1 = 0.7
    p1 = 1
    beta1 = p1 * np.log(n) / n
    alpha1 = np.square((np.sqrt(p1) + np.sqrt(I1))) * np.log(n) / n
    #p_star = max((1 - I1 * n  / 2) * np.log(n) / (gamma * m), 2 * np.log(m) / n)

    #HSBM parameter
    I2 = 2.8 #2^(d-1) for exact recovery
    p2 = 1
    beta2 = p2 * np.log(n) / (comb_compute(n-1, 2))
    alpha2 = np.square((np.sqrt(p2) + np.sqrt(I2))) * np.log(n) / (comb_compute(n-1, 2))

    i_1 = ((1 - I1/k - I2/(k*k)) * np.log(n) / (gamma * m)) / np.square(np.sqrt(1-theta)-np.sqrt(theta))
    i_2 = (k * np.log(m) / n) / np.square(np.sqrt(1-theta)-np.sqrt(theta))
    # print(i_1, i_2)

    # p_star = max(i_1, i_2)
    p_star = 0.2

    # Ground_Rating = [[1, 2, 1],[1, 1, 2]]
    rating_vetor_1_gt = np.random.randint(1, 3, (1, m))
    rating_vetor_2_gt = np.copy(rating_vetor_1_gt)
    rating_vetor_3_gt = np.copy(rating_vetor_1_gt)

    for i in range(int(m * gamma)):
        if rating_vetor_2_gt[0, i] == 1:
            rating_vetor_2_gt[0, i] += 1
        else:
            rating_vetor_2_gt[0, i] += -1

    for i in range(int(m * (1 - gamma*1.5)), m):
        if rating_vetor_3_gt[0, i] == 1:
            rating_vetor_3_gt[0, i] += 1
        else:
            rating_vetor_3_gt[0, i] += -1

    Ground_Rating = np.concatenate((rating_vetor_1_gt, rating_vetor_2_gt, rating_vetor_3_gt), axis=0)
    Ground_Truth = GroundTruth(Ground_Rating, n, m, k, k2)
    gt_idx = np.zeros(n)
    gt_idx[int(n / 3): int(2 * n / 3)] = 1
    gt_idx[int(2*n / 3): n] = 2

    adj_g, inc_g = generate_SBM(n, alpha1, beta1, gt_idx)

    sc1 = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=20)
    sc1.fit(adj_g)
    idx1 = sc1.labels_

    from sklearn.metrics import adjusted_rand_score
    score = adjusted_rand_score(gt_idx, idx1)

    inc_hg = Generate_3_uni_HSBM(n, alpha2, beta2, gt_idx)


    Iter = 100

    err_HG_prob = np.zeros((20, Iter))
    err_G_prob = np.zeros((20, Iter))
    for i1 in range(Iter):
        err_HG = np.zeros(20)
        err_G = np.zeros(20)
        for j1 in range(20):

            p = p_star*(0.1*(j1+1))

            if p > 1:
                p = 1
            # print('p:', p)

            U = MatrixGeneration(p, Ground_Rating, k, k2, theta, q, n, m)

            c_1 = c_estimate_G(idx1, inc_g, n, theta)
            c_2 = c_estimate_HG(idx1, inc_hg, n, theta)

            idx_temp = idx1
            idx_temp_ma = idx1

            numiter = 10
            for k1 in range(numiter):
                out, idx_new, r1, r2, r3 = local_refinement(U, idx_temp, adj_g, c_1)
                out_ma, idx_new_2, r1, r2, r3 = local_refinement_new(U, idx_temp, adj_g, c_1, inc_hg, c_2)

                idx_temp_ma = idx_new_2
                idx_temp = idx_new

                # score = adjusted_rand_score(gt_idx, idx_new)
                err_mat = MAE(out, Ground_Truth)
                err_mat_ma = MAE(out_ma, Ground_Truth)
                # print("HG:", err_mat_ma, "G:", err_mat, 'idx_acc:', score)


            err_HG[j1] = err_mat_ma
            err_G[j1] = err_mat
            # print("idx_acc:", score)
            # print("HG:",err_mat_ma, "G:",err_mat)
        err_HG_prob[:,i1] = err_HG
        err_G_prob[:,i1] = err_G
    err_HG_prob_A = np.where(err_HG_prob == 0, 1, 0)
    err_G_prob_A = np.where(err_G_prob == 0, 1, 0)
    prob_HG = err_HG_prob_A.sum(axis=1) / err_HG_prob.shape[1]
    prob_G = err_G_prob_A.sum(axis=1) / err_G_prob.shape[1]
    print("error prob of HG:", 1 - prob_HG, "error prob of G:", 1 - prob_G)

    # # plotting
    # plt.figure()
    # x = np.arange(0.0, 2.0, 0.1)
    # # plt.ylim(0, 0.5)
    # plt.plot(x, 1 - prob_HG)
    # plt.plot(x, 1 - prob_G)
    # plt.show()
