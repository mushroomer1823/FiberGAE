import pickle
import tensorflow as tf
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

with open('/data/hyf/swm/model1_data/embeddings/embeddings_train.pkl', 'rb') as f:
    features_list = pickle.load(f)
with open('/data/hyf/swm/model1_data/train_range2.pickle', 'rb') as f:
    datasets = pickle.load(f)
with open('/home/hyf/gae-master/gae/classification/fiber_bundle_ids.pkl', 'rb') as f:
    bundle_ids = pickle.load(f)
with open('/home/hyf/gae-master/gae/classification/fiber_bundle_names.pkl', 'rb') as f:
    bundle_names = pickle.load(f)

# print(bundle_ids, len(bundle_ids))
# print(bundle_names, len(bundle_names))
features = np.array([])
labels = np.array([])
for i in range(len(features_list)):
    if i == 0:
        features = features_list[i]
        labels = datasets['labels'][i]
    else:
        features = np.concatenate((features, features_list[i]), axis=0)
        labels = np.concatenate((labels, datasets['labels'][i]), axis=0)
print(labels, labels.shape)
labels_list = np.argmax(labels, axis=1)
for i in range(len(labels_list)):
    temp = bundle_ids[labels_list[i]]
    labels_list[i] = temp
labels_array = np.array(labels_list).reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
labels = encoder.fit_transform(labels_array)
print(labels, labels.shape)

np.random.seed(42)
indices = np.random.permutation(features.shape[0])
features = features[indices, :]
labels = labels[indices, :]

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
num_epochs = 20
batch_size = 64 

initial_learning_rate = 0.001
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                           decay_steps=10000, decay_rate=0.96, staircase=True)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

inputs, targets, logits = build_classifier(input_size, hidden_size1, hidden_size2, output_size)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=33)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # features = np.random.rand(8000, 32)
    # labels = np.eye(800)[np.random.randint(0, 800, size=8000)]

    print(features.shape, labels.shape)
    
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_features = X_train[i:i+batch_size]
            batch_labels = y_train[i:i+batch_size]
            _, batch_loss = sess.run([optimizer, loss], feed_dict={inputs: batch_features, targets: batch_labels})
            
        print("Epoch [{}/{}], Batch Loss: {:.4f}".format(epoch+1, num_epochs, batch_loss))
    
    saver = tf.train.Saver()
    saver.save(sess, '/data/hyf/swm/model1_data/classification_models/classifier_model_normal_lr0.001_44c')
    print("saved model!")

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    batch_features = X_test
    batch_labels = y_test
    batch_acc = sess.run(accuracy, feed_dict={inputs: batch_features, targets: batch_labels})

    print("Test acc: {:.4f}".format(batch_acc))
