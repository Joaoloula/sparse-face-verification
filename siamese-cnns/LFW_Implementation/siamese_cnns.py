import tensorflow as tf
from read_database import read_database
import random
import pickle
import time
from datetime import datetime

# Setting global variables
database_number = 1
max_steps = 20000
images_per_batch = 100
energy_threshold = 0.5

# Setting up train and test sets
face_tensors = pickle.load(open("dictionnary.p", "rb"))
(train_set, test_set) = read_database(database_number)

sess = tf.InteractiveSession()

# INFERENCE
# Placeholders for two images and labels
x1 = tf.placeholder("float", shape=[None, 64, 64])
x2 = tf.placeholder("float", shape=[None, 64, 64])
y_ = tf.placeholder("float", shape=[None])

# Functions for creating weight, bias, convolution and max-pooling


def weight_variable(form, name):
    initial = tf.truncated_normal(form, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(form, name):
    initial = tf.constant(0.1, shape=form)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


# Unused for the moment, fused convolutional and sampling layer implementation
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# Model: Siamese CNNs + energy module
# CNN Architecture: 2 fused conv and sampling layers + 2 fully-connected layers

# Reshape image placeholders
x1_image = tf.reshape(x1, [-1, 64, 64, 1])
x2_image = tf.reshape(x2, [-1, 64, 64, 1])
y = tf.reshape(y_, [1, -1])

# Weights and biases
W_conv1 = weight_variable([6, 6, 1, 5], 'W_conv1')
b_conv1 = bias_variable([5], 'b_conv1')

W_conv2 = weight_variable([6, 6, 5, 14], 'W_conv2')
b_conv2 = bias_variable([14], 'b_conv2')

W_fc1 = weight_variable([16 * 16 * 14, 60], 'W_fc1')
b_fc1 = bias_variable([60], 'b_fc1')

W_fc2 = weight_variable([60, 40], 'W_fc2')
b_fc2 = bias_variable([40], 'b_fc2')

# Layer ops
h1_conv1 = tf.sigmoid(conv2d(x1_image, W_conv1) + b_conv1)
h2_conv1 = tf.sigmoid(conv2d(x2_image, W_conv1) + b_conv1)

h1_conv2 = tf.sigmoid(conv2d(h1_conv1, W_conv2) + b_conv2)
h2_conv2 = tf.sigmoid(conv2d(h2_conv1, W_conv2) + b_conv2)

h1_pool2_flat1 = tf.reshape(h1_conv2, [-1, 16*16*14])
h1_fc1 = tf.sigmoid(tf.matmul(h1_pool2_flat1, W_fc1)+b_fc1)

h2_pool2_flat1 = tf.reshape(h2_conv2, [-1, 16*16*14])
h2_fc1 = tf.sigmoid(tf.matmul(h2_pool2_flat1, W_fc1)+b_fc1)

h1_fc2 = tf.sigmoid(tf.matmul(h1_fc1, W_fc2)+b_fc2)
h2_fc2 = tf.sigmoid(tf.matmul(h2_fc1, W_fc2)+b_fc2)

normal_out1 = tf.transpose(tf.div(tf.transpose(h1_fc2), tf.reduce_sum(
                           tf.abs(h1_fc2), reduction_indices=1)))
normal_out2 = tf.transpose(tf.div(tf.transpose(h2_fc2), tf.reduce_sum(
                           tf.abs(h2_fc2), reduction_indices=1)))

# LOSS
# Normalized absolute difference
absolute_difference = tf.abs(tf.div(tf.sub(normal_out1, normal_out2), 2))
energy = tf.reduce_sum(absolute_difference, reduction_indices=1, name='energy')
loss = tf.reduce_sum(tf.abs(tf.sub(energy, y)))

# SUMMARIES
tensors = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
tensornames = ['W_conv1', 'b_conv1', 'W_conv2', 'b_conv2', 'W_fc1', 'b_fc1',
               'W_fc2', 'b_fc2']
for i in range(8):
    tf.histogram_summary(tensornames[i], tensors[i], name=tensornames[i])

# Record loss in summary
tf.scalar_summary('loss', loss)
merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/siamese_cnns', sess.graph_def)

# TRAINING
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess.run(tf.initialize_all_variables())

for step in xrange(max_steps):
    start_time = time.time()
    batch = random.sample(train_set, images_per_batch)
    # train_set: list with [[name1, name2], label]
    [X1, X2, Y] = [[], [], []]
    for entry in batch:
        X1.append(face_tensors.get(entry[0][0]+'.pgm'))
        X2.append(face_tensors.get(entry[0][1]+'.pgm'))
        Y.append(entry[1])
    # Outputting the loss
    _, loss_value, summary_str = sess.run([train_step, loss, merged_summary_op],
                                          feed_dict={x1: X1, x2: X2, y_: Y})
    duration = time.time()-start_time
    if step % 100 == 0:
        format_str = '%s: step %d, loss = %.2f, (%.3f sec/batch)'
        print format_str % (datetime.now(), step, loss_value, float(duration))
    summary_writer.add_summary(summary_str, step)
