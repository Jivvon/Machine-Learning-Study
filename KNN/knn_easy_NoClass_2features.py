#!/usr/bin/env python
# coding: utf-8

# In[175]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# In[176]:


train_data = np.loadtxt('./iris_train.csv', delimiter=",", dtype=np.float32)
test_data = np.loadtxt('./iris_test.csv', delimiter=",", dtype=np.float32)


# In[177]:


x_train, y_train = train_data[:,:-1], train_data[:,-1] # 훈련 데이터의 특징값(x)과 정답(y)을 분리
x_test, y_test = test_data[:,:-1], test_data[:,-1] # 테스트 데이터의 특징값(x)과 정답(y)을 분리


# In[178]:


# 특징들의 연관성을 한눈에 보기 위해 pandas의 DataFrame으로 데이터를 바꾼 후 scatter_matrix를 실행한다
iris_dataframe = pd.DataFrame(x_train, columns=["feature1","feature2","feature3","feature4"])


# In[179]:


# 특징들의 연관성을 한눈에 보기 위해 pandas의 scatter_matrix를 실행한다
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8)


# ### feature 3과 feature 4에 따라 겹치는 부분이 가장 적게 비교적 잘 구분되는 것을 알 수 있다.
# 
# 각 클래스에 따라 feature 3과 feature 4의 평균값을 구한다

# In[180]:


np.unique(y_train, return_counts=True) # 총 3가지 모두 40개씩 있다.


# In[181]:


x_train_1 = train_data[train_data[:,-1] == 1][:,2:4] # 훈련 데이터를 클래스 별로 나누고 feature 3과 4만 추출한다.
x_train_2 = train_data[train_data[:,-1] == 2][:,2:4]
x_train_3 = train_data[train_data[:,-1] == 3][:,2:4]


# In[182]:


x_mean_1 = np.mean(x_train_1, axis=0) # 각 클래스의 feature 3과 4의 평균값을 구한다.
x_mean_2 = np.mean(x_train_2, axis=0)
x_mean_3 = np.mean(x_train_3, axis=0)


# In[183]:


x_test = test_data[:,2:4] # 테스트 데이터에서 feature 3과 4만 추출한다.


# In[184]:


# 유클리디안 거리
# 각 클래스의 feature 3과 4를 평균과 빼고 각 feature를 제곱 후 더해서 제곱근을 한다.
x_test_1 = np.sqrt(np.sum(np.power(x_test - x_mean_1,2), axis=1))
x_test_2 = np.sqrt(np.sum(np.power(x_test - x_mean_2,2), axis=1))
x_test_3 = np.sqrt(np.sum(np.power(x_test - x_mean_3,2), axis=1))


# In[185]:


# 각 거리의 최소가 되는 인덱스를 구하면 포함되는 클래스를 찾을 수 있다.
predict = np.argmin(np.vstack((x_test_1,x_test_2,x_test_3)).T, axis=1)+1.0


# In[186]:


"예측 정확도 : {}%".format((len(y_test[y_test == predict]) / len(y_test)) * 100)


# ### 예측 정확도가 100.0%으로 나와서 다른 feature를 선택하여 위 과정을 반복해보았다.

# In[187]:


x_train_1 = train_data[train_data[:,-1] == 1][:,:2] # 훈련 데이터를 클래스 별로 나누고 feature 1과 2만 추출한다.
x_train_2 = train_data[train_data[:,-1] == 2][:,:2]
x_train_3 = train_data[train_data[:,-1] == 3][:,:2]

x_mean_1 = np.mean(x_train_1, axis=0) # 각 클래스의 feature 1과 2의 평균값을 구한다.
x_mean_2 = np.mean(x_train_2, axis=0)
x_mean_3 = np.mean(x_train_3, axis=0)

x_test = test_data[:,:2] # 테스트 데이터에서 feature 1과 2만 추출한다.

# 유클리디안 거리
# 각 클래스의 feature 3과 4를 평균과 빼고 각 feature를 제곱 후 더해서 제곱근을 한다.
x_test_1 = np.sqrt(np.sum(np.power(x_test - x_mean_1,2), axis=1))
x_test_2 = np.sqrt(np.sum(np.power(x_test - x_mean_2,2), axis=1))
x_test_3 = np.sqrt(np.sum(np.power(x_test - x_mean_3,2), axis=1))

# 각 거리의 최소가 되는 인덱스를 구하면 포함되는 클래스를 찾을 수 있다.
predict = np.argmin(np.vstack((x_test_1,x_test_2,x_test_3)).T, axis=1)+1.0

"예측 정확도 : {}%".format((len(y_test[y_test == predict]) / len(y_test)) * 100)


# ### feature 1과 2를 선택했을 땐 정확도가 86.6%로 나타나는 것을 확인할 수 있다.

# In[ ]:




