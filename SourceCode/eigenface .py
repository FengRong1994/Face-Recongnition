
# coding: utf-8

# In[1]:


import scipy.io as scio
import numpy as np
import cv2
from numpy import linalg as la
import random
import os
#calculate the recognition parameters
def RecogPara(selecthr=0.99):
    avgImg=np.mean(train_set,1)
    print(len(avgImg))
    diffTrain=(train_set.T-avgImg.T).T
    #print(np.shape(diffTrain))
    
    eigvals,eigVects=la.eig(np.dot(diffTrain,diffTrain.T))
    eigSortIndex=np.argsort(-eigvals)
    for i in range(len(train_set[0])):
        if(eigvals[eigSortIndex[:i]]/eigvals.sum()).sum()>=selecthr:
            eigSortIndex=eigSortIndex[:i]
            #print(i)
            break
    #eigSortIndex=np.delete(eigSortIndex,[0,1,2])
    conVects=eigVects[:,eigSortIndex]
    #print(np.shape(conVects))
    return avgImg, conVects,diffTrain
#judge the input test image
def judge(judgeImg,FaceVector,avgImg,diffTrain,train_num):
    #diff=judgeImg.T-avgImg
    diff=(judgeImg.T-avgImg.T).T
    #print(np.shape(diff))
    weiVec=np.dot(FaceVector.T,diff)
    #print(np.shape(weiVec))
    #weiVec=np.dot(FaceVector.T,diff)
    res=0
    resVal=np.inf
    for i in range(38*train_num):
        TrainVec=np.dot(FaceVector.T,diffTrain[:,i])
        #print(TrainVec)
        if np.sum(np.square(weiVec-TrainVec))<resVal:
        #if(np.array(weiVec-TrainVec)**2).sum()<resVal:
            res=i
            #print(i)
            resVal=np.sum(np.square(weiVec-TrainVec))
            #resVal=(np.array(weiVec-TrainVec)**2).sum()
    return train_label[res]
#read file
datafile='YaleB_32x32.mat'
data=scio.loadmat(datafile)
face=np.array(data['fea'])
label=np.array(data['gnd'])
#change the train_num here
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
#print(2414-train_num*38)
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


# In[2]:


avgImg,FaceVector,diffTrain=RecogPara(selecthr=0.99)
count=0
for i in range(len(test_set[0])):
    a=judge(test_set[:,i],FaceVector,avgImg,diffTrain,train_num)
    b=test_label[i]
    print(a,b)
    
    if a==b:
        count+=1
        print(judge(test_set[:,i],FaceVector,avgImg,diffTrain,train_num),test_label[i])
        #print(count)


# In[9]:


print(count/len(test_label))

