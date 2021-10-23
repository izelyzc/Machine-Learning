#!/usr/bin/env python
# coding: utf-8

# # Homework 03: Nonparametric Regression
# ## İzel Yazıcı
# ### April 19, 2021

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


# In[2]:


# read data into memory
data_set = np.genfromtxt("hw03_data_set.csv", delimiter = ",",skip_header=1)

# get x and y values
x_data = data_set[:,0]
y_data = data_set[:,1]

# get number of classes and number of samples
N = data_set.shape[0]


# In[3]:


#print(N,x_data,y_data)


# In[4]:


x_data=np.array(x_data)
y_data=np.array(y_data)


# In[5]:


point_colors = np.array(["red", "green"])
plt.figure(figsize = (10, 6))
plt.plot(x_data, y_data, "g.", markersize = 10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# ### 3. Divide the data set into two parts by assigning the first 150 data points to the training set and the remaining 122 data points to the test set.

# In[6]:


#Train set
x_train=x_data[:150]
y_train=y_data[:150]

#Test set
x_test=x_data[-122:]
y_test=y_data[-122:]


# In[7]:


print(len(y_test),len(x_test),len(x_train),len(y_train))


# ### 4. Learn a regressogram by setting the bin width parameter to 0.37 and the origin parameter to 1.5. Draw training data points, test data points, and your regressogram in the same figure. Your figure should be similar to the following figure.

# In[42]:


bin_width = 0.37
#origin = 1.5
minimum_value = 1.5
maximum_value = +6
data_interval = np.linspace(minimum_value, maximum_value, 601)


# ### Regressogram

# In[43]:


plt.figure(figsize = (15, 8))
plt.scatter(x = x_train, y = y_train, c = 'b', label = 'training')
plt.scatter(x = x_test, y = y_test, c = 'r', label = 'test')
plt.legend(loc='upper left', borderaxespad=0.5, fontsize = 'large')
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")

#origin 
minimum_value = 1.5
maximum_value= 6
bin_width = 0.37


left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)
p_hat = np.asarray([np.average(y_train[((left_borders[b] < x_train) & (x_train <= right_borders[b]))]) for b in range(len(left_borders))])
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")    
plt.show()

right_borders


# ### 5. Calculate the root mean squared error (RMSE) of your regressogram for test data points.

# In[44]:


rmse = np.asarray([(np.sum(np.square(p_hat[b] - (y_test[(left_borders[b] < x_test) & (x_test <= right_borders[b])]))))for b in range(len(left_borders))]) 
rmse= np.nan_to_num(rmse)
rmse= np.sqrt(np.sum(rmse) / len(x_test))
print(f'Running Regressogram => RMSE is {rmse} when h is {bin_width}')


# ### Mean Smoother

# In[58]:


bin_width = 0.37

#p_hat2 = np.asarray([np.mean(y_train[(np.abs((x - x_train)/bin_width) < 1)])for x in data_interval]) 
#p_hat2 = np.asarray([np.mean(y_train[np.logical_not(np.isnan((np.abs((x - x_train)/bin_width) < 1)) for x in data_interval)]) 
#p_hat2 = np.asarray([np.mean(y_train[(np.abs([(x - x_train[i])/bin_width for i in range(x_train.shape[0])]) < 1)]) for x in data_interval]) 

p_hat2 = np.zeros(601)
counter = 0
for x in data_interval:
    p_hat2[counter] = np.sum((((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width)))*y_train) / np.sum(((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width)))
    counter += 1


plt.figure(figsize = (15, 8))
plt.scatter(x = x_train, y = y_train, c = 'b', label = 'training')
plt.scatter(x = x_test, y = y_test, c = 'r', label = 'test')
plt.legend(loc='upper left', borderaxespad=0.5, fontsize = 'large')
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
#plt.header("h=0.37")
plt.plot(data_interval, p_hat2, "k-")
plt.show()


# ### Calculate the RMSE of your running mean smoother for test data points.

# In[65]:


origin=1.5
error = [(y_test[i] - p_hat2[int((x_test[i]-origin)*x_test.shape[0])])**2 for i in range(len(x_test))]
error= np.nan_to_num(error)
rmse = np.sqrt(np.sum(error) / len(x_test))
print("Mean Smoother => RMSE is", rmse, " when h is", bin_width)


# ## Kernel Smooother

# In[48]:


bin_width = 0.37
ku = [np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)) for x in data_interval]
p_hat3 = np.asarray(([np.sum(y_train * (1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * ((x - x_train)**2 / bin_width**2)))) for x in data_interval]))  / ku 

plt.figure(figsize = (15, 8))
plt.scatter(x = x_train, y = y_train, c = 'b', label = 'training')
plt.scatter(x = x_test, y = y_test, c = 'r', label = 'test')
plt.legend(loc='upper left', borderaxespad=0.5, fontsize = 'large')
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(data_interval, p_hat3, "k-")
plt.show()



# ### RMSE of Kernel Smoother

# In[63]:


origin = 1.5
error = [(y_test[i] - p_hat3[int((x_test[i]-origin)*x_test.shape[0])])**2 for i in range(len(x_test))]
error= np.nan_to_num(error)
rmse = np.sqrt(np.sum(error) / len(x_test))
print("Kernel Smoother => RMSE is", rmse, " when h is", bin_width)

