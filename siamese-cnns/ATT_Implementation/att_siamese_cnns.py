# Benchmarks: 2.8 EER after 8 hours of training for random dataset
# 22 EER after 8 hours for split dataset!!!
import tensorflow as tf
import random
import pickle
import time
from datetime import datetime
from eer import eer
# import numpy as np

# Setting global variables
# max_steps = 20000
pairs_per_batch = 100
A = 1.7159
B = 2./3.

# Setting up train and test sets
[train_set, test_set] = pickle.load(open("train_test_split", "rb"))

sess = tf.InteractiveSession()

# INFERENCE
# Placeholders for two images and labels
x1 = tf.placeholder("float", shape=[None, 64, 64])
x2 = tf.placeholder("float", shape=[None, 64, 64])
y_ = tf.placeholder("float", shape=[None])

# Functions for creating weight, bias, convolution and max-pooling


def weight_variable(form, name):
    initial = tf.truncated_normal(form, stddev=.05)
    return tf.Variable(initial, name=name)


def bias_variable(form, name):
    initial = tf.truncated_normal(form, stddev=.05)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Unused for the moment, fused convolutional and sampling layer implementation
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# Model: Siamese CNNs + energy module
# CNN Architecture: 2 fused conv and sampling layers + 2 fully-connected layers

# Normalize images between -1 and 1
x1_flat = tf.reshape(x1, [-1, 64*64])
x2_flat = tf.reshape(x2, [-1, 64*64])
x1_maxes = tf.reduce_max(x1_flat, reduction_indices=1, keep_dims=True)
x1_mins = tf.reduce_min(x1_flat, reduction_indices=1, keep_dims=True)
x2_maxes = tf.reduce_max(x2_flat, reduction_indices=1, keep_dims=True)
x2_mins = tf.reduce_min(x2_flat, reduction_indices=1, keep_dims=True)
x1_normalized = tf.add(
                       tf.mul(
                                 tf.sub(x1_flat, x1_mins),
                                 2 * tf.sub(x1_maxes, x1_mins)
                                ),
                       -1)
x2_normalized = tf.add(
                       tf.mul(
                                 tf.sub(x2_flat, x2_mins),
                                 2 * tf.sub(x2_maxes, x2_mins)
                                ),
                       -1)


# Reshape image and label placeholders
x1_image = tf.reshape(x1_normalized, [-1, 64, 64, 1])
x2_image = tf.reshape(x2_normalized, [-1, 64, 64, 1])
y = tf.reshape(y_, [1, -1])

# Weights and biases
W_conv1 = weight_variable([6, 6, 1, 5], 'W_conv1')
b_conv1 = bias_variable([5], 'b_conv1')

W_conv2 = weight_variable([6, 6, 5, 14], 'W_conv2')
b_conv2 = bias_variable([14], 'b_conv2')

W_conv3 = weight_variable([6, 6, 14, 60], 'W_conv3')
b_conv3 = bias_variable([60], 'b_conv3')

W_fcl = weight_variable([16*16*60, 40], 'W_fc2')
b_fcl = bias_variable([40], 'b_fc2')

# Layer ops
h1_conv1 = A * tf.tanh(B * (conv2d(x1_image, W_conv1) + b_conv1))
h2_conv1 = A * tf.tanh(B * (conv2d(x2_image, W_conv1) + b_conv1))

maxconv1_1 = max_pool_2x2(h1_conv1)
maxconv1_2 = max_pool_2x2(h2_conv1)

h1_conv2 = A * tf.tanh(B * (conv2d(maxconv1_1, W_conv2) + b_conv2))
h2_conv2 = A * tf.tanh(B * (conv2d(maxconv1_2, W_conv2) + b_conv2))

maxconv2_1 = max_pool_2x2(h1_conv2)
maxconv2_2 = max_pool_2x2(h2_conv2)

h1_conv3 = A * tf.tanh(B * (conv2d(maxconv2_1, W_conv3)+b_conv3))
h2_conv3 = A * tf.tanh(B * (conv2d(maxconv2_2, W_conv3)+b_conv3))

h1_flat = tf.reshape(h1_conv3, [-1, 16*16*60])
h2_flat = tf.reshape(h2_conv3, [-1, 16*16*60])

h1_fcl = A * tf.tanh(B * (tf.matmul(h1_flat, W_fcl)+b_fcl))
h2_fcl = A * tf.tanh(B * (tf.matmul(h2_flat, W_fcl)+b_fcl))

# LOSS
# Normalized absolute difference
absolute_difference = tf.abs(tf.sub(h1_fcl, h2_fcl))
energy = tf.reduce_sum(absolute_difference, reduction_indices=1, name='energy')
loss1 = (2/40) * tf.mul(tf.sub(1., y), tf.square(energy))
loss2 = 2 * 40 * tf.mul(y, tf.exp((-2.7726/40) * energy))
loss = tf.reduce_sum(loss1 + loss2)


# SUMMARIES
tensors = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fcl, b_fcl]
tensornames = ['W_conv1', 'b_conv1', 'W_conv2', 'b_conv2', 'W_conv3', 'b_conv3',
               'W_fcl', 'b_fcl']
for i in range(8):
    tf.histogram_summary(tensornames[i], tensors[i], name=tensornames[i])

# Record loss in summary
tf.scalar_summary('loss', loss)
merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/att_siamese_cnns', sess.graph_def)

# TRAINING
decay_rate = 0.7
learning_rate = tf.Variable(1e-4)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess.run(tf.initialize_all_variables())

step = 1
while 1:
    start_time = time.time()
    batch = random.sample(train_set, pairs_per_batch)
    # train_set: list with [[face1, label1], [face2, label2]]
    [X1, X2, Y] = [[], [], []]
    for i in range(pairs_per_batch):
        X1.append(batch[i][0][0])
        X2.append(batch[i][1][0])
        Y.append(int(batch[i][0][1] != batch[i][1][1]))
    # Outputting the loss
    _, loss_value, summary_str, energy_value = sess.run(
        [train_step, loss, merged_summary_op, energy],
        feed_dict={x1: X1, x2: X2, y_: Y})
    duration = time.time()-start_time
    if step % 100 == 0:
        format_str = '%s: step %d, loss = %.2f, (%.3f sec/batch)'
        print format_str % (datetime.now(), step, loss_value, float(duration))
        # Energy and predictions debugging
        # print energy_value
        # threshold = [i >= 0.5 for i in energy_value]
        # hits_misses = [int(threshold[i] == Y[i]) for i in range(len(Y))]
        # print np.where(np.asarray(hits_misses) == 0)[0]
        summary_writer.add_summary(summary_str, step)
    if step % 1000 == 0:
        # Decay learning rate up to 10^-6
        if learning_rate > 1e-6:
            learning_rate *= decay_rate
        # Evaluate with test set
        [X1, X2, Y] = [[], [], []]
        for i in range(len(test_set)):
            X1.append(test_set[i][0][0])
            X2.append(test_set[i][1][0])
            Y.append(int(test_set[i][0][1] != test_set[i][1][1]))
        energy_value = sess.run([energy], feed_dict={x1: X1, x2: X2, y_: Y})
        error = eer(energy_value[0], Y)
        print error
    step += 1
