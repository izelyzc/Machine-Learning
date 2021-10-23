# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:39:54 2021

@author: izely
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import pandas as pd
from numpy import hstack
from numpy.random import normal
from numpy.linalg import inv 
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from numpy.linalg import inv                    
import matplotlib.pyplot as plt                
from scipy.stats import multivariate_normal     
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_spd_matrix

# read data into memory
X = np.genfromtxt("hw05_data_set.csv", delimiter = ",",skip_header=1)
initial_centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter = ",",skip_header=0)

# get number of classes, number of samples, and number of features
K= initial_centroids.shape[0]
N = X.shape[0]
D = X.shape[1]
# plot data points 
plt.figure(figsize = (10, 10))
plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = 'black')

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)

initial_memberships=update_memberships(initial_centroids,X)

plt.scatter(X[:,0],X[:,1],c=initial_memberships)
plt.scatter(initial_centroids[:,0],initial_centroids[:,1],c="red",marker="s")
plt.show

initial_covariances = []
for i in range(K):
    cov=np.cov(X[initial_memberships==i],rowvar=False)
    initial_covariances.append(cov) 
initial_covariances=np.array(initial_covariances)    


priors_probs = []
for i in range(K):
    priors_probs.append(np.count_nonzero(initial_memberships == i)/ len(initial_memberships)) 

colors = ['tab:blue', 'tab:orange', 'tab:green', 'red', 'yellow']
eps=1e-8

# GMM for 100 steps
for step in range(100):

  # visualize the learned clusters
  if step % 1 == 0:
    plt.figure(figsize=(12,int(8)))
    plt.title("Iteration {}".format(step))
    
    # ell = Ellipse(xy=(m[0],m[1]),
    #           width=w, height=h,
    #           angle=theta, color='black', ls='--')

    # ell.set_facecolor('none')
    # ax.add_artist(ell)
    axes = plt.gca()
    
    likelihood = []
    for j in range(K):
      likelihood.append(multivariate_normal.pdf(x=X, mean=initial_centroids[j], cov=initial_covariances[j]))
    likelihood = np.array(likelihood)
    predictions = np.argmax(likelihood, axis=0)
    
    for c in range(K):
      pred_ids = np.where(predictions == c)
      plt.scatter(X[pred_ids[0],0], X[pred_ids[0],1], color=colors[c], alpha=0.2, edgecolors='none', marker='s')
    
    plt.scatter(X[...,0], X[...,1], facecolors='none', edgecolors='grey')
    
    for j in range(K):
      plt.scatter(initial_centroids[j][0], initial_centroids[j][1], color=colors[j])

    plt.show()

  likelihood = []
  # Expectation step
  for j in range(K):
    likelihood.append(multivariate_normal.pdf(x=X, mean=initial_centroids[j], cov=initial_covariances[j]))
  likelihood = np.array(likelihood)
  assert likelihood.shape == (K, len(X))
    
  b = []
  # Maximization step 
  for j in range(K):
    # use the current values for the parameters to evaluate the posterior
    # probabilities of the data to have been generanted by each gaussian
    b.append((likelihood[j] * priors_probs[j]) / (np.sum([likelihood[i] * priors_probs[i] for i in range(K)], axis=0)+eps))

    # updage mean and variance
    initial_centroids[j] = np.sum(b[j].reshape(len(X),1) * X, axis=0) / (np.sum(b[j]+eps))
    initial_covariances[j] = np.dot((b[j].reshape(len(X),1) * (X - initial_centroids[j])).T, (X - initial_centroids[j])) / (np.sum(b[j])+eps)

    # update the probabilities
    priors_probs[j] = np.mean(b[j])
    
    # assert initial_covariances.shape == (K, X.shape[1], X.shape[1])
    # assert initial_centroids.shape == (K, X.shape[1])