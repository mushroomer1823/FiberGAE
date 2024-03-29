#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 22:23:42 2024

@author: hyf
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from dipy.io.streamline import load_tck
import beta_process

def trainingData():
    # 加载 全脑纤维 和 单个swm 对应的tck文件
    streams = load_tck('/data/swm/testTCK/downsample_1k.tck', '/data/swm/testTCK/dwi.nii.gz')
    cluster = load_tck('/data/swm/testTCK/cluster_1.tck', '/data/swm/testTCK/dwi.nii.gz')
    
    # 将加载的 streamlines 转换为数组形式
    streamlines_array = streams.streamlines
    cluster_array = cluster.streamlines
    
    # 分别计算全脑纤维和聚类中每一根纤维的余弦系数表示
    betas_whole = beta_process.get_betas(streamlines_array)
    betas_single = beta_process.get_betas(cluster_array)
    
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


def build_autoencoder(input_shape):
    model = models.Sequential()
    
    # 编码器部分
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))

    # 解码器部分
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    
    model.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    model.add(layers.Reshape(input_shape))

    return model

array_shape = (30, 1000)
trainset = trainingData()
# 创建 Autoencoder 模型
autoencoder = build_autoencoder(input_shape=array_shape)

# 编译模型
autoencoder.compile(optimizer='adam', loss='mse')

# 将所有数据用于训练
autoencoder.fit(trainset, trainset, epochs=100, batch_size=32, verbose=1)