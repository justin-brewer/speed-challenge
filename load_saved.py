import numpy as np
import tensorflow as tf
import cv2
from time import sleep
#########################
def mse(p, y):
    return np.mean(np.power((p-y), 2))
#########################
label_prec = 8
keep_prob = 1.0

num_hidden_1 = 1800
num_hidden_2 = 500
num_hidden_3 = 250
w_std = 0.02

data = np.float32(np.load('data/test_10798x2400_fzm_5avg.npy'))

d_size = data.shape[1]
speeds = np.float32(np.load('data/train_labels_20400.npy')).reshape(20400, 1)
one_hot_labels = np.load('data/train_labels_20400_one_hot_8xprec.npy')
num_labels = one_hot_labels.shape[1]

num_samples = data.shape[0]

l_filter = np.float32(
    np.arange(num_labels)).reshape(num_labels, 1) / label_prec

graph = tf.Graph()

with graph.as_default():
    keep_prob = tf.placeholder(tf.float32, name='keepProb')
    batch_size = tf.placeholder(tf.int32, name='batch_size')

    # Input data.
    tf_valid_dataset = tf.constant(data)

    # Weights
    layer1_weights = tf.Variable(tf.truncated_normal(
      [d_size, num_hidden_1], stddev=w_std))
    layer1_biases = tf.Variable(tf.zeros([num_hidden_1]))

    layer2_weights = tf.Variable(tf.truncated_normal(
      [num_hidden_1, num_hidden_2], stddev=w_std))
    layer2_biases = tf.Variable(
        tf.constant(1.0, shape=[num_hidden_2]))

    layer3_weights = tf.Variable(tf.truncated_normal(
      [num_hidden_2, num_hidden_3], stddev=w_std))
    layer3_biases = tf.Variable(
        tf.constant(1.0, shape=[num_hidden_3]))

    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden_3, num_labels], stddev=w_std))
    layer4_biases = tf.Variable(
        tf.constant(1.0, shape=[num_labels]))

    saver = tf.train.Saver()

  # Model.
    def model(data, keep_prob, batch_size):
        lc1 = tf.matmul(data, layer1_weights) + layer1_biases
        h1 = tf.nn.relu(lc1)

        lc2 = tf.matmul(h1, layer2_weights) + layer2_biases
        h2 = tf.nn.relu(lc2)

        lc3 = tf.matmul(h2, layer3_weights) + layer3_biases
        h3 = tf.nn.relu(lc3)

        lc4 = tf.matmul(h3, layer4_weights) + layer4_biases
        logits = tf.matmul(tf.nn.softmax(lc4), l_filter)

        return logits

    model_prediction = model(
        tf_valid_dataset, keep_prob, tf_valid_dataset.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    saver.restore(session, "saved_models/relu_model_1800x500x250_0.70_1199_1500.ckpt")

    train_predictions = model_prediction.eval({keep_prob:1.0})
    f_name = 'predictions/test_predictions_70_1199.npy'
    np.save(f_name, train_predictions)
