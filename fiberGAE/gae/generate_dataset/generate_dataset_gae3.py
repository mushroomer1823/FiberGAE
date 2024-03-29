#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/02/2024

@author: hyf
"""
import numpy as np
from dipy.io.streamline import load_tck
import beta_process
import os
import pickle
import networkx as nx
import random
from scipy.sparse import csr_matrix
import h5py
from sklearn.preprocessing import OneHotEncoder
from dipy.segment.metric import mdf

def trainingData():
    count = 0
    dataset = {}        # the dict to be final saved
    feat_list = []      # not csr! cosine coefficient features for all samples
    adj_list = []       # not csr! adjacency matrix
    label_all_list = [] # labels of all fibers for all samples
    graph_list = []     # graph lists, whose length is equal to the number of samples
    adj_csr_list = []   # adjacency matrix in csr matrix
    feat_csr_list = []  # cosine coefficients in csr matrix
    count = 0

    datapath = "/data/hyf/atlas/train/"
    for folderpath, _, filelist in os.walk(datapath, topdown=False):
        print(folderpath)
        if folderpath == "/data/hyf/atlas/train/D_4": continue
        if folderpath == "/data/hyf/atlas/train/F_3": continue
        dataset = {}
        feat_array = np.array([])
        label_list = []
        fiber_list = []

        for index in range(1, 801):
            filename = str(index) + ".tck"
            if filename in filelist:
                filepath = os.path.join(folderpath, filename)
                cluster = load_tck(filepath, '/media/UG5/Atlas/MNI/mni_icbm152_t1_tal_nlin_asym_09a_brain.nii')
                '''
                try:
                    cluster = load_tck(filepath, '/media/UG5/Atlas/MNI/mni_icbm152_t1_tal_nlin_asym_09a_brain.nii')
                except Exception as e:
                    print("an error occured:", e)
                '''
                cluster_array = cluster.streamlines
                # print(cluster_array)
                if len(cluster_array) <= 10:
                    print("length is:", len(cluster_array), "not enough fibers, start repeating fibers")
                    num_to_copy = 10 - len(cluster_array)
                    
                    while num_to_copy > 5:
                        elements_to_copy = cluster_array
                        cluster_array.extend(elements_to_copy)
                        num_to_copy = 10 - len(cluster_array)

                    elements_to_copy = cluster_array[:num_to_copy]
                    cluster_array.extend(elements_to_copy)
                    print("length after repeating: ", len(cluster_array))
                    continue

                sequence = np.random.choice(len(cluster_array), size=10, replace=False)
                cluster_array = cluster_array[sequence]
                # print(len(cluster_array), cluster_array)
                for i in range(len(cluster_array)):
                    fiber_list.append(cluster_array[i])
                
                betas = beta_process.get_betas(cluster_array)
                # print(betas.shape)

                if feat_array.size == 0:
                    feat_array = betas
                else:
                    feat_array = np.concatenate((feat_array, betas), axis=0)
                label_list.extend([index]*10)
                print("success read ", folderpath, index)
                
                # print(feat_array.shape)
                # print(len(label_list))

        if feat_array.size == 0:
            continue

        distance_array = np.zeros((len(fiber_list), len(fiber_list)))
        for i in range(len(fiber_list)):
            for j in range(i+1, len(fiber_list)):
                fiber1 = fiber_list[i]
                fiber2 = fiber_list[j]
                mdf_distance = mdf(fiber1, fiber2)
                distance_array[i, j] = mdf_distance
                distance_array[j, i] = mdf_distance

        flat_array = np.sort(distance_array.flatten())
        threshold_index = int(0.01 * len(flat_array))
        threshold_value = flat_array[threshold_index]
        print("the threshold is:", threshold_value)

        # indices = np.random.choice(feat_array.shape[0], size=10000, replace=False)
        # print(indices)
        # feat_array = feat_array[indices]
        
        label_array = np.array(label_list) - 1
        label_array = np.array(label_array).reshape(-1, 1)
        categories = [range(0, 800)]
        encoder = OneHotEncoder(categories=categories, sparse=False)
        label_onehot = encoder.fit_transform(label_array)
        print("label:", label_array.shape)
        print("onehot:", label_onehot.shape)

        G = nx.Graph()
        for i in range(feat_array.shape[0]):
            feature1 = feat_array[i]
            G.add_node(i, feature = feature1)
       
        print("number of nodes:", G.number_of_nodes())
        print("label_list length:", len(label_list))
        print("shape of feature array:", feat_array.shape)
        for idx_i, (node_i, data_i) in enumerate(G.nodes(data=True)):
            feature_i = data_i['feature']
            for idx_j, (node_j, data_j) in enumerate(G.nodes(data=True)):
                if node_i != node_j:
                    # feature_j = data_j['feature']
                    # print("index:", idx_i, idx_j)
                    '''
                    # 方法1：同一个聚类中的节点全部连接，不同聚类间的节点没有连接
                    if label_list[idx_i] == label_list[idx_j]:
                        G.add_edge(node_i, node_j)
                    
                    # 方法2：根据距离大小判断是否连接，选取一个范围，保留距离在这个范围之间的连接
                    feature_j = data_j['feature']
                    distance = np.linalg.norm(feature_i - feature_j)
                    if distance <= 30:
                        G.add_edge(node_i, node_j, weight = distance)
                    
                    if distance >= 20 and distance <= 40:
                        G.add_edge(node_i, node_j, weight=distance)
                    elif distance >= 70 and distance <= 90:
                        G.add_edge(node_i, node_j, weight=distance)
                    '''
                    # 方法3：根据dipy的mdf方法计算纤维之间的距离，只保留前1%短的纤维
                    distance = distance_array[idx_i, idx_j]
                    if distance <= threshold_value:
                        G.add_edge(node_i, node_j, weight=distance)
            
        print("number of edges: ", G.number_of_edges())

        adj_matrix = nx.to_numpy_array(G)
        adj_csr = csr_matrix(adj_matrix)
        feature_csr = csr_matrix(feat_array)
        graph_list.append(G)

        feat_list.append(feat_array)
        adj_list.append(adj_matrix)
        label_all_list.append(label_onehot)
        graph_list.append(G)
        feat_csr_list.append(feature_csr)
        adj_csr_list.append(adj_csr)

        count += 1
        # print("count = ", count)
        # if count == 2: break
    
    dataset['features'] = feat_list
    dataset['adj'] = adj_list
    dataset['labels'] = label_all_list
    # dataset['graph'] = graph_list
    # dataset['features_csr'] = feat_csr_list
    # dataset['adj_csr'] = adj_csr_list

    picklepath = "/data/hyf/swm/model4_data/train_range5.pickle"
    h5path = "/home/hyf/gcn-master/gcn/train.h5"

    with open(picklepath, 'wb') as file:
        pickle.dump(dataset, file, protocol=2)

    print("have saved pickle file!")
    '''
    with h5py.File(h5path, 'w') as f:
        f.create_dataset('features', data=dataset['features'])
        f.create_dataset('labelss', data=dataset['labels'])
        f.create_dataset('graph', data=dataset['graph'])
        f.create_dataset('features_csr', data=dataset['features_csr'])
        f.create_dataset('adj_csr', data=dataset['adj_csr'])
    '''
trainingData()

