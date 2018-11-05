
# coding: utf-8

# In[3]:


from sklearn.datasets import load_breast_cancer


# In[4]:


cancer = load_breast_cancer()


# In[5]:


print("cancer.keys(): \n{}".format(cancer.keys()))


# In[6]:


print(cancer.data)


# In[7]:


# print the names of the four features
print(cancer.feature_names)


# In[8]:


print(cancer.DESCR)


# In[10]:


print(cancer.target)


# In[11]:


print(cancer.data)


# In[12]:


print(cancer.target_names)


# In[13]:


print(type(cancer.data))
print(type(cancer.target))


# In[14]:


print(cancer.data.shape)


# In[15]:


print(cancer.target.shape)


# In[16]:


# store feature matrix in "X"
X = cancer.data

# store response vector in "y"
y = cancer.target


# In[20]:


print(X.shape)
print(y.shape)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier


# In[22]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[23]:


print(knn)


# In[24]:


knn.fit(X, y)


# In[26]:


from IPython.display import IFrame


# In[29]:




