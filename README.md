# FiberGAE
use Graph Auto Encoder for fiber bundle classification

genarate the graph dataset:
/fiberGAE/gae/genarate_dataset/generate_dataset_gae.py
build edges between nodes from the same cluster

train the GAE network:
/fiberGAE/gae/train.py

get embeddings for classification network:
/fiberGAE/gae/test.py

train classification network:
/fiberGAE/gae/classification/net.py

get classification results on testing dataset:
/fiberGAE/gae/classification/test.py
