
# coding: utf-8

# In[11]:


import scipy.io as scio
import numpy as np
import cv2
from numpy import linalg as la
import random
import os
#read file
datafile='YaleB_32x32.mat'
data=scio.loadmat(datafile)
face=np.array(data['fea'])
label=np.array(data['gnd'])
#change train number here
train_num=50
print(train_num)
list_0=[];
list_total=[];
list_index=[];
for i in range(1,39):
    list_total.append(0)
    list_index.append(0)
for i in range(0,2414):
    list_0.append(label[i][0])
for i in range(1,39):
    list_total[i-1]=list_0.count(i)
temp=0
for i in range(1,38):
    temp+=list_total[i-1]
    list_index[i]=temp
face=face.T
label=label.T
train_set=np.zeros((1024,train_num*38))
train_label=[]
test_set=np.zeros((1024,(2414-train_num*38)))

test_label=[]
train_index=[]
for i in range(0,37):
    train_index.append(random.sample(range(list_index[i],list_index[i+1]-1),train_num))
#print(train_index)
train_index.append(random.sample(range(list_index[37],2414),train_num))

train_index=np.array(train_index).flatten()
temp1=0
temp2=0
#print(len(label[0])-1)
for i in range(len(label[0])):
    if i in train_index:
        train_set[:,temp1]=face[:,i]
        train_label.append(label[:,i])
        temp1+=1
    else:
        #print(temp2)
        test_set[:,temp2]=face[:,i]
        test_label.append(label[:,i])
        temp2+=1
train_set=train_set.T
test_set=test_set.T
train_label=np.array(train_label)
test_label=np.array(test_label)
train_label=train_label.astype(np.int32)
test_label=test_label.astype(np.int32)


# In[12]:


fisherf=cv2.face.FisherFaceRecognizer_create()
fisherf.train(train_set,train_label)
result=[]
dist=[]
cnt=0
for i in range(len(test_set)):
    temp=test_set[i]
    res_temp,dist_temp=fisherf.predict(test_set[i])
    result.append(res_temp)
    dist.append(dist_temp)
    print(res_temp,test_label[i])
    if res_temp==test_label[i]:
        cnt+=1
print(cnt/len(test_set))

