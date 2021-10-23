# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 21:07:08 2021

@author: izely
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.spatial as spa


# read data into memory
X = np.genfromtxt("hw06_data_set.csv", delimiter = ",",skip_header=1)

# get number of classes, number of samples, and number of features
N = X.shape[0]
D = X.shape[1]
# plot data points 
plt.figure(figsize = (10, 10))
plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = 'black')

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

## B Adjacency/ Connectivity Matrix

def pairwise_distances(X, Y):

    #Calculate distances from every point of X to every point of Y actualy X

    #start with all zeros
    distances = np.empty((len(X),len(Y)), dtype='float')

    #compute adjacencies
    for i in range(len(X)):
        for j in range(len(Y)):
            distances[i, j] = np.linalg.norm(X[i]-Y[j])
                
    return distances


## conectivity matrix
𝛿 = 1.25
B=pairwise_distances(X,X)

# plot data points 

G = nx.DiGraph() 
for i in range(B.shape[0]): 
 for j in range( B.shape[1]): 
   if B[i][j] <= 𝛿: 
      G.add_edge(i,j) 
      

nx.draw( G )
plt.xlabel("x1")
plt.ylabel("x2")
plt.figure(figsize = (10, 10))
plt.show()

## D Matrix number of neighbors 
def nearest_neighbor_graph(X):

    X = np.array(X)

    # for smaller datasets use sqrt(#samples) as n_neighbors. max n_neighbors = 10
    n_neighbors = min(int(np.sqrt(X.shape[0])), 10)

    #calculate pairwise distances
    A = pairwise_distances(X, X)

    #sort each row by the distance and obtain the sorted indexes
    sorted_rows_ix_by_dist = np.argsort(A, axis=1)

    #pick up first n_neighbors for each point (i.e. each row)
    #start from sorted_rows_ix_by_dist[:,1] because because sorted_rows_ix_by_dist[:,0] is the point itself
    nearest_neighbor_index = sorted_rows_ix_by_dist[:, 1:n_neighbors+1]

    #initialize an nxn zero matrix
    W = np.zeros(A.shape)

    #for each row, set the entries corresponding to n_neighbors to 1
    for row in range(W.shape[0]):
        W[row, nearest_neighbor_index[row]] = 1

    #make matrix symmetric by setting edge between two points if at least one point is in n nearest neighbors of the other
    for r in range(W.shape[0]):
        for c in range(W.shape[0]):
            if(W[r,c] == 1):
                W[c,r] = 1


    return W


# Laplacian Matrix
def compute_laplacian(W):

    # 𝐋symetric = I-D^{-1/2} B D^{-1/2}

    # calculate row sums
    d = W.sum(axis=1)

    #create degree matrix
    D = np.diag(d)
    
 
    #H = np.sum(G[0])
    for id, x in enumerate(W):
        D[id][id] = np.sum(x)
    
    I = np.identity(len(W))
    
    # 𝐋symetric = I-D^{-1/2} B D^{-1/2}
    
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    L = I - np.dot(D_inv_sqrt, W).dot(D_inv_sqrt)

    return L

# Eigen Vectors
𝑅 = 5
def get_eigvecs(L, R):

    eigvals, eigvecs = np.linalg.eig(L)
    # sort eigenvalues and select R smallest values - get their indices
    ix_sorted_eig = np.argsort(eigvals)[:R]

    #select k eigenvectors corresponding to R-smallest eigenvalues
    return eigvecs[:,ix_sorted_eig]

# K=5
def k_means_pass(X, K, n_iters):
    
    #initial centroids
    centers = [85,129,167,187,270] 

    for iteration in range(n_iters):
        #calculate distances for every point in X to each of the k centers
        distance_pairs = pairwise_distances(X, centers)

        #assign label to each point - index of the centroid with smallest distance
        labels = np.argmin(distance_pairs, axis=1)
        new_centers = [np.nan_to_num(X[labels == i].mean(axis=0)) for i in range(K)]
        new_centers = np.array(new_centers)

        #check for convergence of the centers
        if np.allclose(centers, new_centers):
            break

        #update centers for next iteration
        centers = new_centers


    return centers, labels

def cluster_distance_metric(X, centers, labels):
    
    # Metric to evaluate how close points in the clusters are to their centroid
    # Returns sum of all distances of points to their corresponding centroid

    return sum(np.linalg.norm(X[i]-centers[labels[i]]) for i in range(len(labels)))

def k_means_clustering(X, K):
    solution_labels = None
    current_metric = None

    #run k_means pass, so that each pass starts at a different initial random point.
    for pass_i in range(10):
        #perform a pass
        centers, labels = k_means_pass(X, K, 100)

        #calculate distance metric for the solution
        new_metric = cluster_distance_metric(X, centers, labels)
        #keep track of the smallest metric and its solution
        if current_metric is None or new_metric < current_metric:
            current_metric = new_metric
            solution_labels = labels

    return solution_labels


def spectral_clustering(X, K):

    #create weighted adjacency matrix
    W = nearest_neighbor_graph(X)

    #create unnormalized graph Laplacian matrix
    L = compute_laplacian(W)

    #create projection matrix with first k eigenvectors of L
    E = get_eigvecs(L, K)

    #return clusters using k-means on rows of projection matrix
    y = k_means_clustering(E, K)
    return np.ndarray.tolist(y)

spec_clusters= spectral_clustering(X, 5)