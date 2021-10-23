#!/usr/bin/env python
# coding: utf-8

# # HW02: Discrimination by Regression
# ## Izel Yazici
# ### March 27, 2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def safelog(x):
    return(np.log(x + 1e-100))

# read data into memory
# images pixels data
X = np.genfromtxt("hw02_images.csv", delimiter = ",")
# target 
y_truth = np.genfromtxt("hw02_labels.csv", delimiter = ",").astype(int)

W = np.genfromtxt("initial_W.csv", delimiter = ",")
w0 = np.genfromtxt("initial_w0.csv", delimiter = ",")
# get number of samples and number of categorical classes
K = np.max(y_truth)
N = X.shape[0]

# one-of-K encoding
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth - 1] = 1


X.shape
W.shape

#Train set
X_train=X[0:500,:]
Y_train=Y_truth[0:500,:]

#Test set
X_test=X[0:-500,:]
Y_test=Y_truth[0:-500,:]

X.shape

Y_test.shape

Y_truth.shape

#print(X)
#print(y_truth)
print(W)
print(w0)
print(Y_truth)
print(N)
print(K)


# ### Learn a discrimination by regression algorithm using the sigmoid function for this multiclass classification problem. You can use the following learning parameters.

# define the sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))

draw_seq = np.linspace(-10, +10, 2001)
plt.figure(figsize = (10, 6))
plt.plot(draw_seq, 1 / (1 + np.exp(-draw_seq)), "r-")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.show()

sig=sigmoid(X, W, w0)


# ## Gradient Function

# define the gradient functions
def gradient_W(X, Y_truth, Y_predicted):
    return(np.asarray([-np.sum(np.repeat((Y_truth[:,c] - Y_predicted[:,c])[:, None], X.shape[1], axis = 1) * X, axis = 0) for c in range(K)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(Y_truth - Y_predicted, axis = 0))


# ### Learn a discrimination by regression algorithm using the sigmoid function for this multiclass classification problem. You can use the following learning parameters.

# ## Algorithm Parameters

# set learning parameters
eta = 0.0001
epsilon = 1e-3
max_it = 500


# ## Iterative Algorithm

# learn W and w0 using gradient descent
# learn W and w0 using gradient descent
iteration = 1
objective_values = []
while 1:
    Y_predicted = sigmoid(X_train, W, w0)

    objective_values = np.append(objective_values,np.sum((Y_train-Y_predicted)**2)/2)
    #-np.sqrt(((Y_train - Y_predicted) ** 2).mean()))
    #-np.sum(((Y_train)**2-(Y_predicted))**2))
    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(X_train, Y_train, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_train, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break
    if iteration >= max_it:
        break    

    iteration = iteration + 1
print(W)
print(w0)

iteration
#500

# plot objective function during iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


print(W.shape)
print(X.shape)
print(Y_truth.shape)
print(Y_predicted.shape)
print(X_train.shape)
print(Y_train.shape)


X_train.shape
X_train

# ## Convergence

# plot objective function during iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# ## Training Performance

k = np.max(Y_train)
n = X_train.shape[0]

# calculate confusion matrix
y_train=y_truth[0:500]
y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix_train = pd.crosstab(y_predicted, y_train, rownames = ['y_pred'], colnames = ['y_train_truth'])
print(confusion_matrix_train)

# calculate confusion matrix
y_test=y_truth[500:1000]
y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix_test = pd.crosstab(y_predicted, y_test, rownames = ['y_pred'], colnames = ['y_test_truth'])
print(confusion_matrix_test)

#_test_truth   1   2   3   4   5
#_pred                                                      
#              80  5   5   0   3
#              0  82   0   0   0
#              12  4  97   0   3
#              0   0   0  103  1
#              7   0   3   0  95


Y_train


# ## Visualization

# evaluate discriminant function on a grid
x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
for c in range(K):
    discriminant_values[:,:,c] = W[0, c] * x1_grid + W[1, c] * x2_grid + w0[0, c]

A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
D = discriminant_values[:,:,3]
E = discriminant_values[:,:,4]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C
discriminant_values[:,:,3] = D
discriminant_values[:,:,4] = E



plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize = 10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize = 10)
plt.plot(X[y_truth == 4, 0], X[y_truth == 4, 1], "b.", markersize = 10)
plt.plot(X[y_truth == 5, 0], X[y_truth == 5, 1], "b.", markersize = 10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,3], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,4], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,0], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,3], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,2] - discriminant_values[:,:,4], levels = 0, colors = "k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# ### Plotting Data


# plot data points generated
plt.figure(figsize = (20, 20))
plt.plot(X[:,0], y_truth[:,0]==1, "r.", markersize = 10)
plt.plot(X[:,1], y_truth[:,0]==3, "b.", markersize = 10)
#plt.plot(images[:,2], labels[:,0], "g.", markersize = 10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# # Alternative

# ### E({wi,wi0}i|X)= ∥r −y ∥ = (ri −yi ) 2t 2ti


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, x, y):
    h = sigmoid(x @ theta)
    m = len(y)
    cost = 1 / m * np.sum(
        -y * np.log(h) - (1 - y) * np.log(1 - h)
    )
    grad = 1 / m * ((y - h) @ x)
    return cost, grad


# ## Discrimination by Regression


def fit(x, y, max_iter=max_it, alpha=0.1):
    x = np.insert(x, 0, 1, axis=1)
    thetas = []
    classes = np.unique(y)
    costs = np.zeros(max_iter)

    for c in classes:
        # one vs. rest binary classification
        binary_y = np.where(y == c, 1, 0)
        
        theta = np.zeros(x.shape[1])
        for epoch in range(max_iter):
            costs[epoch], grad = cost(theta, x, binary_y)
            theta += alpha * grad
            
        thetas.append(theta)
    return thetas, classes, costs




fit(X,y_truth,max_it,eta)





max_iter=500
alpha=0.1
thetas = []
classes = np.unique(y_truth)
costs = np.zeros(max_iter)

for c in classes:
    # one vs. rest binary classification
    binary_y = np.where(y_truth == c, 1, 0)
    
    theta = np.zeros(X.shape[1])
    for epoch in range(max_iter):
        costs[epoch], grad = cost(theta, X, binary_y)
        theta += alpha * grad
            
        thetas.append(theta)


# ## Predict Function




def predict(classes, thetas, x):
    x = np.insert(x, 0, 1, axis=1)
    preds = [np.argmax(
        [sigmoid(xi @ theta) for theta in thetas]
    ) for xi in x]
    return [classes[p] for p in preds]


# In[142]:


T=thetas
#print(T)
predict(3,T,X)





