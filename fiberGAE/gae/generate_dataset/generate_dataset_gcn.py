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
        dataset = {}
        feat_array = np.array([])
        label_list = []

        for index in range(1, 801):
            filename = str(index) + "_edited.tck"
            if filename in filelist:
                filepath = os.path.join(folderpath, filename)
                try:
                    cluster = load_tck(filepath, '/media/UG5/Atlas/MNI/mni_icbm152_t1_tal_nlin_asym_09a_brain.nii')
                except Exception as e:
                    print("an error occured:", e)
                cluster_array = cluster.streamlines
                # print(cluster_array)
                if len(cluster_array) <= 50:
                    print("not enough fibers")
                    continue
                sequence = np.random.choice(len(cluster_array), size=50, replace=False)
                cluster_array = cluster_array[sequence]
                # print(len(cluster_array))
                betas = beta_process.get_betas(cluster_array)
                # print(betas.shape)

                # dataset['feat'] = np.concatenate((dataset['deat'], betas), axis=0)
                if feat_array.size == 0:
                    feat_array = betas
                else:
                    feat_array = np.concatenate((feat_array, betas), axis=0)
                label_list.extend([index]*50)
                print("success read ", folderpath, index)
                
                # print(feat_array.shape)
                # print(len(label_list))

        if feat_array.size == 0:
            continue
        indices = np.random.choice(feat_array.shape[0], size=10000, replace=False)
        # print(indices)
        feat_array = feat_array[indices]
        
        label_array = np.array(label_list)[indices] - 1
        label_array = np.array(label_array).reshape(-1, 1)
        categories = [range(0, 800)]
        encoder = OneHotEncoder(categories=categories, sparse=False)
        label_onehot = encoder.fit_transform(label_array)
        # print("label:", label_array.shape)
        # print("onehot:", label_onehot.shape)

        G = nx.Graph()
        for i in range(10000):
            feature1 = feat_array[i]
            G.add_node(i, feature = feature1)

        for node_i, data_i in G.nodes(data=True):
            feature_i = data_i['feature']
            for node_j, data_j in G.nodes(data=True):
                if node_i != node_j:
                    feature_j = data_j['feature']
                    distance = np.linalg.norm(feature_i - feature_j)
                    if distance >= 20 and distance <= 40:
                        G.add_edge(node_i, node_j, weight=distance)
                    elif distance >= 70 and distance <= 90:
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

    picklepath = "/home/hyf/gcn-master/gcn/train_range1.pickle"
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

