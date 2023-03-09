# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:51:17 2021

@author: ariken
"""
import os
import SimpleITK as sitk
import tensorflow as tf
import numpy as np
import random
import copy

def file_name(file_dir):   
   label=[] 
   img = []
   path_list = os.listdir(file_dir)
   path_list.sort() 
   for filename in path_list:
       if 'nii' in filename:
           if 'label' in filename:
               label.append(os.path.join(filename))
           else:
               img.append(os.path.join(filename))
   return img, label

def saveNii(cc, pos):
    trainlabelList[:], trainList[:], trainImgName[:], trainLabelName[:] = zip(*cc)
    for index in range(len(trainList)):
        sitk.WriteImage(sitk.GetImageFromArray(trainList[index]), pos + trainImgName[index])
        sitk.WriteImage(sitk.GetImageFromArray(trainlabelList[index]), pos + trainLabelName[index])
        
def meanForAll(x):
    meanList = []
    for img in x:
        meanList.append(np.mean(img))
    return np.mean(meanList)

def stdForAll(x):
    temp = copy.deepcopy(x)
    for img in temp[1:]:        
        temp[0] = np.concatenate((temp[0], img), axis = -1)    
    return np.std(temp[0])

def preprocessing(train, label, downSamplingFactor = 8):
    resultTrain = []
    resultLabel = []
    mean = meanForAll(train)
    std = stdForAll(train)
    for index in range(len(train)):
        #train[index] = (train[index] - np.mean(train[index]))/np.std(train[index])
        train[index] = (train[index] - mean)/std
    return resultTrain, resultLabel
file = 'ALL/'

trainImgName, trainLabelName = file_name(file)
trainList, trainlabelList = [], []

'''
Flip = RandomFlipLayer(flip_axes=[1])
Flip.randomise()

Scaling = RandomSpatialScalingLayer(min_percentage=-10, max_percentage=10)
Scaling.randomise()

Rotating = RandomRotationLayer()
Rotating.init_uniform_angle((-10, 10))
Rotating.randomise()
'''
pretrainList = []
TTraining = []
FinetuneF = []
test = []

for i in range(len(trainImgName)):
    trainImg = sitk.ReadImage(file + trainImgName[i], sitk.sitkFloat32)
    trainLabelImg = sitk.ReadImage(file + trainLabelName[i], sitk.sitkFloat32)
    trainList.append(tf.transpose(sitk.GetArrayFromImage(trainImg), [2, 1, 0]).numpy())
    trainlabelList.append(tf.transpose(sitk.GetArrayFromImage(trainLabelImg), [2, 1, 0]).numpy())

#mean = meanForAll(trainList[0:115])
#std = stdForAll(trainList[0:115])
#for index in range(len(trainList)):
    #trainList[index] = (trainList[index] - np.mean(trainList[index]))/np.std(trainList[index])
    #trainList[index] = (trainList[index] - mean)/std

cc = list(zip(trainlabelList, trainList, trainImgName, trainLabelName))
random.shuffle(cc)
'''
for i in range(round(len(trainList) * 0.4)):
    pretrainList.append(cc[i])
    
for j in range(round(len(trainList) * 0.2)):
    TTraining.append(cc[i+j+1])
    
for k in range(round(len(trainList) * 0.2)):
    FinetuneF.append(cc[i+j+k+1])
    
for t in range(round(len(trainList) * 0.2)):
    test.append(cc[i+j+k+t+1])
    '''
pretrainList = cc[0:77]
TTraining = cc[77:115]
FinetuneF = cc[115:153]
test = cc[153:192]


saveNii(pretrainList, 'pretrain//')
saveNii(TTraining, 'TTraining//')
saveNii(FinetuneF, 'Validate//')
saveNii(test, 'test//')
    
    
    

    
    
    
    
    
    
    
    
    