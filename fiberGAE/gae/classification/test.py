import pickle
import tensorflow as tf
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# with open('/data/hyf/swm/model1_data/embeddings_test_5subs_3epoches.pkl', 'rb') as f:
with open('/data/hyf/swm/model1_data/embeddings/embeddings_test.pkl', 'rb') as f:
    features_list = pickle.load(f)
# with open('/data/hyf/swm/model1_data/test_5subjects.pickle', 'rb') as f:
with open('/data/hyf/swm/model1_data/test_5subjects.pickle', 'rb') as f:
    datasets = pickle.load(f)
with open('/home/hyf/gae-master/gae/classification/fiber_bundle_ids.pkl', 'rb') as f:
    bundle_ids = pickle.load(f)
with open('/home/hyf/gae-master/gae/classification/fiber_bundle_names.pkl', 'rb') as f:
    bundle_names = pickle.load(f)

print(len(features_list))
print(features_list[0].shape)
features = np.array([])
labels = np.array([])
for i in range(len(features_list)):
    if i == 0:
        features = features_list[i]
        labels = datasets['labels'][i]
    else:
        features = np.concatenate((features, features_list[i]), axis=0)
        labels = np.concatenate((labels, datasets['labels'][i]), axis=0)

labels_list = np.argmax(labels, axis=1)
for i in range(len(labels_list)):
    temp = bundle_ids[labels_list[i]]
    labels_list[i] = temp
labels_array = np.array(labels_list).reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
labels = encoder.fit_transform(labels_array)

'''
np.random.seed(123)
indices = np.random.permutation(features.shape[0])
features = features[indices, :]
labels = labels[indices, :]
'''
print(features.shape, labels.shape)

def build_classifier(input_size, hidden_size1, hidden_size2, output_size):
    inputs = tf.placeholder(tf.float32, shape=[None, input_size])
    targets = tf.placeholder(tf.int32, shape=[None, output_size])
    
    fc1 = tf.layers.dense(inputs, hidden_size1, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, hidden_size2, activation=tf.nn.relu)
    logits = tf.layers.dense(fc2, output_size)
    
    return inputs, targets, logits

input_size = 32
hidden_size1 = 256
hidden_size2 = 512
output_size = 44
learning_rate = 0.03
num_epochs = 200 
batch_size = 64 

inputs, targets, logits = build_classifier(input_size, hidden_size1, hidden_size2, output_size)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '/data/hyf/swm/model1_data/classification_models/classifier_model_normal_lr0.001_44c')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batch_features = features
    batch_labels = labels
    batch_acc, predicted_labels = sess.run([accuracy, logits], feed_dict={inputs: batch_features, targets: batch_labels})
    
    predicted_labels = np.argmax(predicted_labels, axis=1)
    print("Test acc: {:.4f}".format(batch_acc))
    print(predicted_labels, len(predicted_labels))
    
    '''
    labels_list = np.argmax(labels, axis=1)
    unequal_indices = np.where(labels_list != predicted_labels)[0]
    for idx in unequal_indices:
        true_label = labels_list[idx]
        predicted_label = predicted_labels[idx]
        # print("idx: ", idx, "true label: ", true_label, "predicted_label: ", predicted_label)
    
    
    # get the predicted labels (1-800 clusters)
    predicted_names = []
    for label_id in range(len(predicted_labels)):
        cluster_id = predicted_labels[label_id]
        predicted_labels[label_id] = bundle_ids[cluster_id]
        predicted_names.append(bundle_names[cluster_id])
    print(len(predicted_labels), predicted_labels)

    # turn the 800 clusters' labels into form of fiber bundle names
    labels_list = np.argmax(labels, axis=1)
    true_names = []
    for label_id in range(len(labels_list)):
        cluster_id = labels_list[label_id]
        labels_list[label_id] = bundle_ids[cluster_id]
        true_names.append(bundle_names[cluster_id])
    print(len(labels_list), labels_list)

    correct_predictions = np.sum(labels_list == predicted_labels)
    print(correct_predictions)
    accuracy = float(correct_predictions) / float(len(labels_list))
    print(len(labels_list))
    print("fiber bundles acc:", accuracy)
    '''
'''
unequal_indices = np.where(labels_list != predicted_labels)[0]

for idx in unequal_indices:
    true_label = labels_list[idx]
    B
    predicted_label = predicted_labels[idx]
    true_name = true_names[idx]
    predicted_name = predicted_names[idx]
    print("idx: ", idx, "true name: ", true_name, "predicted_name: ", predicted_name)
'''
