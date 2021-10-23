#!/usr/bin/env python
# coding: utf-8

# # Homework 04: Decision Tree Regression
# ## İzel Yazıcı
# ### April 26, 2021

import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",",skip_header=1)

# get x and y values
x_data = data_set[:,0]
y_data = data_set[:,1].astype(int)

# get number of classes, number of samples, and number of features
K = np.max(y_data)
N = data_set.shape[0]
D = data_set.shape[1]

#print(N,x_data,y_data)

x_data=np.array(x_data)
y_data=np.array(y_data)


# ### Divide the data set into two parts by assigning the first 150 data points to the training set and the remaining 122 data points to the test set.


#Train set
X_train=x_data[:150]
y_train=y_data[:150]

#Test set
X_test=x_data[-122:]
y_test=y_data[-122:]


# get numbers of train and test samples
N_train = len(y_train)
N_test = len(y_test)

# ### Tree Inference

# Implement a decision tree regression algorithm using the following pre-pruning rule: If a
# node has 𝑃 or fewer data points, convert this node into a terminal node and do not split
# further, where 𝑃 is a user-defined parameter.


def DecisionTree(P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}
    node_splits = {}
    node_avg_value = {}


    # Put all training instances into the root node
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True

    # Learning algorithm
    while True:
        # Find nodes that need splitting
        split_nodes = [key for key, value in need_split.items() if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # Find best split positions for all nodes and prunning value
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean = np.mean((y_train[data_indices]))
            if len((X_train[data_indices])) <= P:
                is_terminal[split_node] = True
                node_avg_value[split_node] = node_mean
                break
            else:
                is_terminal[split_node] = False
                unique_values = np.sort(np.unique(X_train[data_indices]))
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0, len(split_positions))
                
                for d in range(len(split_positions)):
                    left_indices = data_indices[X_train[data_indices] <= split_positions[d]]
                    right_indices = data_indices[X_train[data_indices] > split_positions[d]]
                    total_error = 0
                    if len(left_indices) > 0:
                        total_error += np.sum((y_train[left_indices] - np.mean(y_train[left_indices]))**2)
                    if len(right_indices) > 0:
                        total_error += np.sum((y_train[right_indices] - np.mean(y_train[right_indices]))**2)
                    split_scores[d] = total_error / (len(left_indices) + len(right_indices)+1)

                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split

                # Create left node using the selected split
                left_indices = data_indices[X_train[data_indices] < best_split]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                # create right node using the selected split
                right_indices = data_indices[X_train[data_indices] >= best_split]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
    return (is_terminal, node_splits, node_avg_value)
                

def Score_DT(point, is_terminal, node_splits, node_avg_value):
    index = 1
    while(True):
        if is_terminal[index] == True:
            return node_avg_value[index]
        else:
            if point < node_splits[index]:
                index = 2*index
            else:
                index = 2*index + 1


# Learn a decision tree by setting the pre-pruning parameter 𝑃 to 25. Draw training data
# points, test data points, and your fit in the same figure.

data_intervals = np.linspace(np.min(X_train), np.max(X_train), 270)

P = 25
is_terminal, node_splits, node_avg_value = DecisionTree(25)

y_pred = [Score_DT(data_intervals[i], is_terminal, node_splits, node_avg_value) for i in range(len(data_intervals))]

#Score_DT(data_intervals[1], is_terminal, node_splits, node_avg_value)
#data_intervals[1]


plt.figure(figsize = (15, 8))
plt.title(f"P = {P}")
plt.scatter(x = X_train, y = y_train, c = 'b', label = 'training')
plt.scatter(x = X_test, y = y_test, c = 'r', label = 'test')
plt.plot(data_intervals, y_pred, "k")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc='upper left', borderaxespad=0.5, fontsize = 'large')
plt.legend()
plt.show()

y_pred = [ Score_DT(X_test[i], is_terminal, node_splits, node_avg_value) for i in range(N_test)]
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print(f"RMSE is {rmse} when P is {P}")

# Learn decision trees by setting the pre-pruning parameter 𝑃 to 5, 10, 15, …, 50. Draw
# RMSE for test data points as a function of 𝑃. Your figure should be similar to the
# following figure.

Ps = np.arange(5,51,5)
y_preds = np.zeros((len(Ps), N_test))
for p in range(len(Ps)):
    is_terminal, node_splits, node_avg_value = DecisionTree(Ps[p])
    y_pred = [ Score_DT(X_test[i], is_terminal, node_splits, node_avg_value) for i in range(N_test)]
    y_preds[p] = y_pred

#np.mean((y_test - y_preds[8])**2)
total_rmse = [np.sqrt(np.mean((y_test - y_preds[i])**2)) for i in range(10)]

print(f"RMSE is {total_rmse} when P is {Ps}")

plt.figure(figsize=(15,8))
plt.scatter(Ps, total_rmse, color="black")
plt.plot(Ps, total_rmse, color="black")
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.show()



# ### Rule Extraction

## extract rules
#terminal_nodes = [key for key, value in is_terminal.items() if value == True]
#for terminal_node in terminal_nodes:
#    index = terminal_node
#    rules = np.array([])
#    while index > 1:
#        parent = np.floor(index / 2)
#        if index % 2 == 0:
#            # if node is left child of its parent
#            rules = np.append(rules, "x{:d} < {:.2f}".format(node_features[parent] + 1, node_splits[parent]))
#        else:
#            # if node is right child of its parent
#            rules = np.append(rules, "x{:d} >= {:.2f}".format(node_features[parent] + 1, node_splits[parent]))
#        index = parent
#    rules = np.flip(rules)
#    print("{} => {}".format(rules, node_avg_value[terminal_node]))