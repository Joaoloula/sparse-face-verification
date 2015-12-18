# DeepID implementation on tensorflow

import tensorflow as tf
import random
import pickle
import time
from datetime import datetime
import numpy as np
from PIL import Image

# Setting global variables
# max_steps = 20000
imgs_per_batch = 500

# Setting up train and test sets
[train_set, test_set] = pickle.load(open("facescrub_labeled", "rb"))

sess = tf.InteractiveSession()

# INFERENCE
# Placeholders for two images and labels
x = tf.placeholder("float", shape=[None, 32, 32])
y_ = tf.placeholder("float", shape=[None, 530])


# Functions for creating weight, bias, convolution and max-pooling
def weight_variable(form, name):
    initial = tf.truncated_normal(form, stddev=.1)
    return tf.Variable(initial, name=name)


def bias_variable(form, name):
    initial = tf.truncated_normal(form, stddev=.1)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def normalize(tensor, size):
    # Local response linearization: parameterers can be tweaked
    return tf.nn.lrn(tensor, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


# Model: Siamese CNNs + energy module
# CNN Architecture: 4 convolution layers with max-pooling in between +
# one DeepID layer + 1 softmax output layer

# Reshape image and label placeholders
x_image = tf.reshape(x, [-1, 32, 32, 1])
y = tf.reshape(y_, [-1, 530])

# Weights and biases
W_conv1 = weight_variable([4, 4, 1, 20], 'W_conv1')
b_conv1 = bias_variable([20], 'b_conv1')

W_conv2 = weight_variable([3, 3, 20, 40], 'W_conv2')
b_conv2 = bias_variable([40], 'b_conv2')

W_conv3 = weight_variable([3, 3, 40, 60], 'W_conv3')
b_conv3 = bias_variable([60], 'b_conv3')

W_conv4 = weight_variable([2, 2, 60, 80], 'W_conv4')
b_conv4 = bias_variable([80], 'b_conv4')

W_deepid_conv3 = weight_variable([8*8*60, 160], 'W_deepid_conv3')
W_deepid_conv4 = weight_variable([4*4*80, 160], 'W_deepid_conv4')
b_deepid = bias_variable([160], 'b_deepid')

W_fcl = weight_variable([160, 530], 'W_fcl')
b_fcl = bias_variable([530], 'b_fcl')

# Layer ops
conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
max_pool1 = max_pool_2x2(conv1)
max_pool1 = normalize(max_pool1, 4)

conv2 = tf.nn.relu(conv2d(max_pool1, W_conv2)+b_conv2)
max_pool2 = max_pool_2x2(conv2)
max_pool2 = normalize(max_pool2, 4)

conv3 = tf.nn.relu(conv2d(max_pool2, W_conv3)+b_conv3)
max_pool3 = max_pool_2x2(conv3)
max_pool3 = normalize(max_pool3, 4)

conv4 = tf.nn.relu(conv2d(max_pool3, W_conv4)+b_conv4)
conv4 = normalize(conv4, 4)

conv3_flat = tf.reshape(conv3, [-1, 8*8*60])
conv4_flat = tf.reshape(conv4, [-1, 4*4*80])

deepid = tf.nn.relu(tf.matmul(conv3_flat, W_deepid_conv3) +
                    tf.matmul(conv4_flat, W_deepid_conv4) +
                    b_deepid)
# deepid = normalize(deepid, 4)

softmax = tf.nn.softmax(tf.matmul(deepid, W_fcl)+b_fcl)

# LOSS
loss = -tf.reduce_sum(tf.log(tf.reduce_sum(tf.mul(softmax, y), 1)))

# ACCURACY
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(softmax, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# SUMMARIES
tensors = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4,
           b_conv4, W_deepid_conv3, W_deepid_conv4, b_deepid, W_fcl, b_fcl]
tensornames = ['W_conv1', 'b_conv1', 'W_conv2', 'b_conv2', 'W_conv3', 'b_conv3',
               'W_conv4', 'b_conv4', 'W_deepid_conv3', 'W_deepid_conv4',
               'b_deepid', 'W_fcl', 'b_fcl']
for i in range(13):
    tf.histogram_summary(tensornames[i], tensors[i], name=tensornames[i])

# Record loss in summary
tf.scalar_summary('loss', loss)
merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/att_siamese_cnns', sess.graph_def)

# TRAINING
decay_rate = 1  # First tests without decay
learning_rate = tf.Variable(0.01)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess.run(tf.initialize_all_variables())

step = 1
while 1:
    start_time = time.time()
    batch = random.sample(train_set, imgs_per_batch)
    [X, Y] = [[], []]
    for i in range(imgs_per_batch):
        image = np.asarray(Image.open(batch[i][0]).convert('1'))
        X.append(image)
        onehot_y = np.zeros(530)
        onehot_y[batch[i][1]] = 1
        Y.append(onehot_y)
    # Outputting the loss
    _, loss_value, summary_str = sess.run([train_step, loss, merged_summary_op],
                                          feed_dict={x: X, y_: Y})
    duration = time.time()-start_time
    if step % 100 == 0:
        format_str = '%s: step %d, loss = %.2f, (%.3f sec/batch)'
        print format_str % (datetime.now(), step, loss_value, float(duration))
        summary_writer.add_summary(summary_str, step)
    if step % 1000 == 0:
        # Decay learning rate up to 10^-6
        if learning_rate > 1e-6:
            learning_rate *= decay_rate
        # Evaluate with test set
        batch = random.sample(test_set, 1000)
        [X, Y] = [[], []]
        for i in range(1000):
            image = np.asarray(Image.open(batch[i][0]).convert('1'))
            X.append(image)
            onehot_y = np.zeros(530)
            onehot_y[batch[i][1]] = 1
            Y.append(onehot_y)
        total_accuracy = sess.run([accuracy], feed_dict={x: X, y_: Y})
        print total_accuracy
    step += 1
