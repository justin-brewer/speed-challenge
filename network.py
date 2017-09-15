import numpy as np
import tensorflow as tf
import timeit
###############################################################
num_hidden_1 = 1800
num_hidden_2 = 500
num_hidden_3 = 250
train_drop = 0.5
num_steps = 1200
print_step = 10
w_std = 0.02
label_prec = 8
train_ratio = 0.6
###############################################################
#37
label_list = ['train_labels_20400_one_hot_8xprec.npy',
                'train_labels_20400.npy',]
#37
one_hot_labels = np.load('data/' + label_list[0])
#37
labels = np.float32(np.load('data/' + label_list[1]))
#37
data_list = ['train_20400x2400_flow_zoom.npy',]
#37
data = np.float32(np.load('data/' + data_list[0]))
num_labels = one_hot_labels.shape[1]
d_size = data.shape[1]
num_samples = data.shape[0]
labels = labels.reshape(num_samples, 1)
np.set_printoptions(precision=2)
l_filter = (np.float32(
    np.arange(num_labels)).reshape(num_labels, 1) / label_prec) + 1
train_i = int(num_samples * train_ratio)
b_size = 1500
###############################################################
rp = np.random.permutation(num_samples)
labels = labels[rp]
data = data[rp,...]
one_hot_labels = one_hot_labels[rp,...]

train_dataset = data[:train_i,...]
train_labels = labels[:train_i]
train_labels_a = one_hot_labels[:train_i,...]

valid_dataset = data[train_i:,...]
valid_labels = labels[train_i:]
valid_labels_a = one_hot_labels[train_i:,...]

def mse(p, y):
    return np.mean(np.power((p-y), 2))
####################_DEF_GRAPH___###########################
graph = tf.Graph()
with graph.as_default():
    keep_prob = tf.placeholder(tf.float32, name='keepProb')
    batch_size = tf.placeholder(tf.int32, name='batch_size')

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(None, d_size), name='train_data')
    tf_train_labels = tf.placeholder(tf.float32,
        shape=(None), name='train_labels')
    tf_valid_dataset = tf.constant(valid_dataset)

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
        pred = tf.matmul(tf.nn.softmax(lc4), l_filter) - 1

        return pred

    # Training computation.
    train_prediction = model(tf_train_dataset, keep_prob, batch_size)

    loss = tf.losses.mean_squared_error(
        labels=tf_train_labels, predictions=train_prediction)

    train_full = model(
        train_dataset, keep_prob, train_dataset.shape[0])

    valid_prediction = model(
        tf_valid_dataset, keep_prob, tf_valid_dataset.shape[0])

    # Optimizer.
    optimizer = tf.train.AdamOptimizer().minimize(loss)
####################__RUN_NET___############################
start = timeit.default_timer()

min_val_mse = 100.0
min_step = 0
min_loss = 0
prev_loss = 10000
conv_timer = 0
conv_timer_stop = 1
conv_crit = 0.000001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        # npboot
        batch_ids = np.random.choice(train_i, b_size)
        batch_data = train_dataset[batch_ids,...]
        batch_labels = train_labels[batch_ids]

        feed_dict = {tf_train_dataset : batch_data,
            tf_train_labels : batch_labels,
            keep_prob:train_drop, batch_size:b_size}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)

        loss_dif = abs(l - prev_loss)

        if loss_dif < conv_crit:
            conv_timer += 1
        if conv_timer == conv_timer_stop:
            break
        prev_loss = l

        if (step % print_step == 0):
            feed_dict['keep_prob'] = 1.0
            print('Minibatch (%d) loss at step %d: %8.6f' %
                (b_size, step, l))
            train_mse = mse(train_full.eval(
                {keep_prob:1.0}), train_labels)
            print('\tMB) MSE: {%4.6f}/{%4.6f}' %
                 (mse(predictions, batch_labels), train_mse))
            val_mse = mse(
                valid_prediction.eval({keep_prob:1.0}), valid_labels)
            print('\tVD) MSE: {%4.6f}' % (val_mse))

            if val_mse < min_val_mse:
                min_val_mse = val_mse
                min_step = step
                min_loss = l

    elapsed = (timeit.default_timer() - start)/60
    e_sec = int((elapsed % 1) * 60)

    print('min) l: %4.6f step: %d Minimum MSE: {%4.6f}' %
          (min_loss, min_step, min_val_mse))
    #37
    model_path = ("saved_models/relu_model_%dx%dx%d_%2.2f_%d_%d.ckpt" %
        (num_hidden_1, num_hidden_2, num_hidden_3, train_ratio, step, b_size))

    # save_path = saver.save(session, model_path)
