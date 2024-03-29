import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import pickle
import os
import h5py
import networkx as nx

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
# from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from gae.single_net_train import get_embeddings

def loadData():
    datapath = "/data/hyf/atlas/train"
    foldername = os.listdir(datapath)
    foldername.remove('H_4')
    fibers = np.array([])
    labels = np.array([])
    for sample in foldername:
        filepath = os.path.join(datapath, sample, 'cos.h5')
        labelpath = os.path.join(datapath, sample, '2000', 'label.h5')
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
        
        for l in range(1, 2001):
            indices = np.where(label == l)
            cfiber = fiber[indices]
            np.random.shuffle(cfiber)
            # print("cfiber: ", cfiber.shape)
            if cfiber.shape[0] == 0: continue

            G = nx.Graph()
            for i in range(cfiber.shape[0]):
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

            value = int(adj_matrix.size*0.9)
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
            
            print("final: ", adj_matrix.shape, feature_matrix.shape)
            print(np.count_nonzero(adj_matrix))

            adj_csr = sp.csr_matrix(adj_matrix)
            feature_csr = sp.csr_matrix(feature_matrix)
            print(adj_csr.shape, feature_csr.shape)
            embeddings = get_embeddings(adj_csr, feature_csr)
            print(embeddings.shape)



    '''
        for index in range(1, 801):
            filename = str(index) + "_edited.tck"
            if filename in filelist:
                filepath = os.path.join(folderpath, filename)
                
        savepath = folderpath + "/cos.h5"
        print(savepath)
        transposed = feat_list_array.T
        
        with h5py.File(savepath, "w") as hf:
            hf.create_dataset("data", data=transposed)
    '''
loadData()

