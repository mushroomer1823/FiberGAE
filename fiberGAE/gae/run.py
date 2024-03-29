import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import pickle
import os
import h5py
import networkx as nx
import random

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.model_selection import train_test_split
from keras import models, layers
import keras

from gae.input_data import load_data
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from gae.single_net_train import get_embeddings

def train():
    datapath = "/data/hyf/atlas/train"
    foldername = os.listdir(datapath)
    foldername.remove('H_4')
    
    fibers = np.array([])
    labels = np.array([])
    
    dataList = []
    labelList = []

    count = 0
    
    for sample in foldername:
        if count == 2: break
        filepath = os.path.join(datapath, sample, 'cos_bigdata.h5')
        labelpath = os.path.join(datapath, sample, '1000', 'label.h5')
        embeddingpath = os.path.join(datapath, sample, '2000', 'embeddings.pkl')
        with open(embeddingpath, 'rb') as f:
            global_embedding = pickle.load(f)
        print(global_embedding.shape)
        with h5py.File(filepath, 'r') as file:
            data = file['data']
            fiber = np.array(data)
            fiber = fiber.T
            if fibers.size == 0:
                fibers = fiber
            else:
                fibers = np.concatenate((fibers, fiber), axis=0)
        with h5py.File(labelpath, 'r') as file:
            data = file['train_data']
            label = np.array(data)
            if labels.shape == 0:
                labels = label
            else:
                labels = np.concatenate((labels, label), axis=0)
        count += 1

        for l in range(1, 1001):
            print("sample", count, "cluster", l)
            indices = np.where(label == l)
            cfiber = fiber[indices]
            np.random.shuffle(cfiber)
            # print("cfiber: ", cfiber.shape)
            if cfiber.shape[0] == 0 or cfiber.shape[0] < 1000:
                print("not enough fibers")
                continue

            G = nx.Graph()
            for i in range(1000):
                feature1 = cfiber[i, :]
                G.add_node(i, feature = feature1)

            for node_i, data_i in G.nodes(data=True):
                feature_i = data_i['feature']
                for node_j, data_j in G.nodes(data=True):
                    if node_i != node_j:
                        feature_j = data_j['feature']
                        distance = np.linalg.norm(feature_i - feature_j)
                        G.add_edge(node_i, node_j, weight=distance)

            adj_matrix = nx.to_numpy_matrix(G)

            value = int(adj_matrix.size*0.25)
            flattened_adj = adj_matrix.flatten()
            flattened_adj.sort()
            threshold_value = flattened_adj[:,value]
            print(threshold_value)
            adj_matrix[adj_matrix > threshold_value] = 0

            feature_matrix = np.array([])
            for node, data in G.nodes(data=True):
                feature = data['feature']
                feature = np.expand_dims(feature, axis=0)
                if feature_matrix.size == 0:
                    feature_matrix = feature
                else:
                    feature_matrix = np.concatenate((feature_matrix, feature), axis=0)
                # print("feature matrix: ", feature_matrix.shape) 
            
            feature_matrix = feature_matrix.reshape(feature_matrix.shape[0], feature_matrix.shape[1]*feature_matrix.shape[2])
            print("final: ", adj_matrix.shape, feature_matrix.shape)
            
            adj_csr = sp.csr_matrix(adj_matrix)
            feature_csr = sp.csr_matrix(feature_matrix)
            # print(adj_csr.shape, feature_csr.shape)
            print(np.count_nonzero(adj_matrix))
            local_embedding = get_embeddings(adj_csr, feature_csr)
            
            embeddings = np.concatenate((global_embedding, local_embedding), axis=0)
            print(embeddings.shape)
            dataList.append(embeddings)
            labelList.append(l-1)

    '''
    for i in range(100):
        array = np.random.rand(3000, 32)
        label = random.randint(0, 999)
        dataList.append(array)
        labelList.append(label)
    '''

    print("length of datalist: ", len(dataList))
    print("length of labellist: ", len(labelList))
    
    dataList = np.stack(dataList)
    # dataList = np.random.rand(100,3000,32)
    label_onehot = keras.utils.to_categorical(labelList, num_classes=1000)
    X_train, X_test, y_train, y_test = train_test_split(dataList, label_onehot, test_size=0.2, random_state=42)
    
    input_layer = layers.Input(shape=(3000, 32))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(1000, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    print("acc: ", accuracy)

    
train()

