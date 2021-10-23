#!/usr/bin/env python
# coding: utf-8

# # Homework 01: Naïve Bayes’ Classifier

# ### Import Libraires

# In[63]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


# ### 2. Import Data

# In[3]:


images_df = pd.read_csv("hw01_images.csv", header = None)
labels_df = pd.read_csv("hw01_labels.csv", header = None)


# In[4]:


print(len(images_df))
print(len(labels_df))


# In[5]:


images_df


# In[6]:


labels_df.head()


# In[7]:


images_df.info()


# In[8]:


labels_df.head()


# In[9]:


np.unique(labels_df)


# ### 3. Split Dataset as Train and Test

# Divide the data set into two parts by assigning the first 200 images to the training set and
# the remaining 200 images to the test set.

# In[10]:


train_images_df=images_df[:200]
test_images_df=images_df[-200:]


# In[11]:


train_labels_df=labels_df[:200]
test_labels_df=labels_df[-200:]


# ### 4. Estimate Mean, Standart Deviation and Prior Probabilities

# 4.Estimate the mean parameters 𝜇̂!,!, 𝜇̂!,#,…, 𝜇̂!,$%&', 𝜇̂#,!, 𝜇̂#,#,…, 𝜇̂#,$%&', the standard
# deviation parameters 𝜎'!,!, 𝜎'!,#,…, 𝜎'!,$%&', 𝜎'#,!, 𝜎'#,#,…, 𝜎'#,$%&', and the prior probabilities
# 𝑃 )
# (𝑦 = 1),𝑃 )(𝑦 = 2) using the data points you assigned to the training set in the previous
# step.

# In[12]:


print(len(train_images_df))
print(len(train_labels_df))


# In[13]:


column_count=len(train_images_df.columns)
print(column_count)


# In[14]:


train_labels_df


# $\widehat{\mu_{c}} = \dfrac{\sum\limits_{i = 1}^{N} x_{i} \mathbb{1}(y_{i} = c)}{\sum\limits_{i = 1}^{N} \mathbb{1}(y_{i} = c)}$

# In[15]:


x=train_images_df.to_numpy()
#y = y.transpose()
#y=train_labels_df.transpose().to_numpy().astype(int)
y = []
for i in range(len(train_labels_df)):
    y.append(int(train_labels_df.transpose()[i]))
y= np.array(y)


# In[77]:


x_test=test_images_df.to_numpy()
#y = y.transpose()
#y=train_labels_df.transpose().to_numpy().astype(int)
y_test = []
for i in range(len(test_labels_df)):
    y_test.append(int(test_labels_df.transpose()[i]))
y_test= np.array(y_test)

# get number of classes and number of samples
K = len(np.unique(y))
N = train_images_df.shape[0]
print(K)
print(N)


# calculate sample means
sample_means=[]  
sample_means
for i in range(column_count):
    sample_means_X1 = np.array([np.mean(x[:,i][y == c ])  for c in range(1, K+1)])
    sample_means.append(sample_means_X1)
    print(sample_means_X1)


# ### Results

# In[25]:


sample_means = np.array(sample_means)


# In[31]:


sample_means[:,0]


# In[32]:


sample_means[:,1]


# $\widehat{P}(y_{i} = c) = \dfrac{\sum\limits_{i = 1}^{N} \mathbb{1}(y_{i} = c)}{N}$

# In[33]:


priors = []
for i in range(1,K+1):
    priors.append(np.count_nonzero(y == i)/ len(y)) 


# In[34]:


priors


# In[35]:


sample_means[0]


# In[53]:


sample_means[200,0]


# In[36]:


x[0]


# In[46]:


# standart deviation [200,1]
np.sqrt(np.mean((x[:,200][y == 1 ] - sample_means[200,1])**2))


# In[69]:


sample_deviations=[]  
sample_deviations
for i in range(column_count):
    standart_deviations = [np.sqrt(np.mean((x[:,i][y == c+1] - sample_means[i,c])**2)) for c in range(K)]
    sample_deviations.append(standart_deviations)
    print(standart_deviations)


# In[67]:


sample_deviations = np.array(sample_deviations)
sample_deviations[0,1]


# In[68]:


#standart_deviations[0]
sample_deviations[:,0]


# $\widehat{\sigma_{c}^{2}} = \dfrac{\sum\limits_{i = 1}^{N} (x_{i} - \widehat{\mu_{c}})^{2} \mathbb{1}(y_{i} = c)}{\sum\limits_{i = 1}^{N} \mathbb{1}(y_{i} = c)}$

# In[75]:


def safelog(x):
    return(np.log(x + 1e-100))


# In[ ]:


def p()


# In[ ]:


## Define NaiveBayes Classifier 
def NVB (x,y,sample_means,sample_deviations,priors):
    g_x=x[i,c]*safelog(p[])
    
    


# In[73]:


# evaluate score functions
score_values = np.stack([np.log(2 * math.pi * sample_deviations[i,c]**2) 
                         (column_count - sample_means[i,c])**2 / sample_deviations[i,c]**2 
                         + np.log(priors[c])
                         for c in range(K)])


# In[74]:


#### Alternative Way

from sklearn.utils.validation import check_X_y, check_array

def NBC(self, X: np.ndarray, y: np.ndarray):
    """ Fit training data for Naive Bayes classifier """

    # not strictly necessary, but this ensures we have clean input
    X, y = check_X_y(X, y)
    n = X.shape[0]

    X_by_class = np.array([X[y == c] for c in np.unique(y)])
    self.prior = np.array([len(X_class) / n for X_class in X_by_class])

    self.word_counts = np.array([sub_arr.sum(axis=0) for sub_arr in X_by_class]) + self.alpha
    self.lk_word = self.word_counts / self.word_counts.sum(axis=1).reshape(-1, 1)

    self.is_fitted_ = True
    return self


# ### Parameters

# In[ ]:





# 5. Calculate the confusion matrix for the data points in your training set using the
# parametric classification rule you will develop using the estimated parameters. Your
# confusion matrix should be similar to the following matrix.

# In[ ]:


# calculate confusion matrix
y_predicted = 1 * (y_predicted > 0.5)
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# 6. Calculate the confusion matrix for the data points in your test set using the parametric
# classification rule you will develop using the estimated parameters. Your confusion
# matrix should be similar to the following matrix.

# In[ ]:




