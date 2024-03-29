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
import h5py

def trainingData():
    datapath = "/data/hyf/atlas/train"
    
    foldername = os.listdir(datapath)
    for folder in foldername:
        folderpath = os.path.join(datapath, folder)
        print(folderpath)
        feat_list_array = np.array([])
        # print(feat_list_array.size)
        filelist = os.listdir(folderpath)
        for index in range(1, 801): 
            filename = str(index) + ".tck"
            if filename in filelist:
                filepath = os.path.join(folderpath, filename)
                try:
                    cluster = load_tck(filepath, '/media/UG5/Atlas/MNI/mni_icbm152_t1_tal_nlin_asym_09a_brain.nii') # 加载单个聚类对应的tck文件
                    cluster_array = cluster.streamlines # 将加载的 streamlines 转换为数组形式
                    
                    betas = beta_process.get_betas(cluster_array) # 分别计算全脑纤维和聚类中每一根纤维的余弦系数表示
                    print(betas.shape)
                    if feat_list_array.size == 0:
                        feat_list_array = betas
                    else:
                        feat_list_array = np.concatenate((feat_list_array, betas), axis=0)
                    # print(feat_list_array.shape)
                    
                    # dataset['feat'] = np.concatenate((dataset['deat'], betas), axis=0)
                    
                except Exception as e:
                    print(f"an error occured: {str(e)}")

        savepath = folderpath + "/cos_bigdata.h5"
        print("savepath:", savepath)
        transposed = feat_list_array.T
        # print(transposed.shape)
        with h5py.File(savepath, "w") as hf:
            hf.create_dataset("data", data=transposed)

trainingData()

