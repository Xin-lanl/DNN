import tensorflow as tf
import numpy as np
import scipy.stats as ss

def parameters_init():
  # initial parameters
  w1 = np.float32(ss.truncnorm.rvs(0, 5e-2, size=[5, 5, 3, 64]))
  b1 = np.zeros(64, dtype='f')
  w2 = np.float32(ss.truncnorm.rvs(0, 5e-2, size=[5, 5, 64, 64]))
  b2 = np.zeros(64, dtype='f')
  w3 = np.float32(ss.truncnorm.rvs(0, 0.04, size=[756, 384]))
  b3 = np.zeros(384, dtype='f')
  w4 = np.float32(ss.truncnorm.rvs(0, 0.04, size=[384, 384]))
  b4 = np.zeros(384, dtype='f')
  w5 = np.float32(ss.truncnorm.rvs(0, 1/384.0, size=[384, 10]))
  b5 = np.zeros(10, dtype='f')
  return (w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)

def parameters_conf(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
  # init parameters
  parameters = {
    'w_conv1': tf.Variable(w1, dtype=tf.float32),
    'w_conv2': tf.Variable(w2, dtype=tf.float32),
    'w_fc1': tf.Variable(w3, dtype=tf.float32),
    'w_fc2': tf.Variable(w4, dtype=tf.float32),
    'w_out': tf.Variable(w5, dtype=tf.float32),
    'b_conv1': tf.Variable(b1, dtype=tf.float32),
    'b_conv2': tf.Variable(b2, dtype=tf.float32),
    'b_fc1': tf.Variable(b3, dtype=tf.float32),
    'b_fc2': tf.Variable(b4, dtype=tf.float32),
    'b_out': tf.Variable(b5, dtype=tf.float32)  
  }
  return parameters

def inference(images, parameters, dropout_prob):
  # conv1 
  net = tf.nn.conv2d(images, parameters['w_conv1'], [1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, parameters['b_conv1'])
  net = tf.nn.relu(net)
  net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
  # conv2
  net = tf.nn.conv2d(net, parameters['w_conv2'], [1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, parameters['b_conv2'])
  net = tf.nn.relu(net)
  net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
  # fc1
  net = tf.reshape(net, [-1, parameters['w_fc1'].get_shape().as_list()[0]])
  net = tf.add(tf.matmul(net, parameters['w_fc1']), parameters['b_fc1'])
  net = tf.nn.relu(net)
  net = tf.nn.dropout(net, 1 - dropout_prob)
  # fc2
  net = tf.add(tf.matmul(net, parameters['w_fc2']), parameters['b_fc2'])
  net = tf.nn.relu(net)
  net = tf.nn.dropout(net, 1 - dropout_prob)
  # output
  net = tf.add(tf.matmul(net, parameters['w_out']), parameters['b_out'])

  return net 

w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = parameters_init()
parameters = parameters_conf(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
# alexnet(x, parameters, 0.8)
dropout_prob = 0.2
pred = inference(x, parameters, dropout_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))