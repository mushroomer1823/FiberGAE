#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 22:23:42 2024

@author: hyf
"""
import numpy as np
from dipy.io.streamline import load_tck
import beta_process
import os
import pickle

def trainingData():
    dataset = {}
    feat_list = []
    label_list = []
    subject_name_list = []

    dataset['label_name'] = [f"cluster_{num:05}" for num in range(1, 801)]

    datapath = "/data/hyf/atlas/val"
    for count, (folderpath, _, filelist) in enumerate(os.walk(datapath)):
        print(count)
        print(folderpath)
        for index in range(1, 801): 
            filename = str(index) + "_edited.tck"
            if filename in filelist:
                filepath = os.path.join(folderpath, filename)
                try:
                    cluster = load_tck(filepath, '/media/UG5/Atlas/MNI/mni_icbm152_t1_tal_nlin_asym_09a_brain.nii') # 加载单个聚类对应的tck文件
                    cluster_array = cluster.streamlines # 将加载的 streamlines 转换为数组形式
                    if len(cluster_array) != 500:
                        print("not enough fibers")
                        continue
                    betas = beta_process.get_betas(cluster_array) # 分别计算全脑纤维和聚类中每一根纤维的余弦系数表示
                    
                    # dataset['feat'] = np.concatenate((dataset['deat'], betas), axis=0)
                    feat_list.append(betas)
                    label_list.extend([index]*500)
                    subject_name_list.extend([count]*500)
                    print("success read ", folderpath, index)
                    print(betas.shape)
                except Exception as e:
                    print(f"an error occured: {str(e)}")

    feat_list_array = np.array(feat_list)
    shape = (feat_list_array.shape[0] * feat_list_array.shape[1], feat_list_array.shape[2], feat_list_array.shape[3])
    dataset['feat'] = feat_list_array.reshape(shape)
    dataset['label'] = np.array(label_list) - 1
    dataset['subject_id'] = np.array(subject_name_list)


    print(len(dataset['feat']))
    print(len(dataset['label']))
    print(len(dataset['label_name']))
    print(len(dataset['subject_id']))

    picklepath = "/home/hyf/TractCloud/TrainData_traindata/val.pickle"
    with open(picklepath, 'wb') as file:
        pickle.dump(dataset, file)

    '''
    # 创建对单个聚类的训练数据集
    fiber_train = np.zeros((len(betas_single), 30, 1000))
    
    for i in range(len(betas_single)):
        fiber_feature = betas_single[i]
        for j in range(1,1000):
            whole_feature = betas_whole[j]
            line = np.concatenate([whole_feature, fiber_feature], axis=0)
            fiber_train[i, :, j] = line[:, 0]
    
    print("shape of training dataset: ", fiber_train.shape)
    return fiber_train
    '''

trainingData()

