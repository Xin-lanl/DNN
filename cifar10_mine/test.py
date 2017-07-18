import tensorflow as tf
import numpy as np
import scipy.stats as ss

def parameters_init():
  # initial parameters
  w1 = np.float32(ss.truncnorm.rvs(0, 5e-2, size=[5, 5, 3, 64]))
  b1 = np.zeros(64, dtype='f')
  w2 = np.float32(ss.truncnorm.rvs(0, 5e-2, size=[5, 5, 64, 64]))
  b2 = np.zeros(64, dtype='f')
  w3 = np.float32(ss.truncnorm.rvs(0, 0.04, size=[2304, 512]))
  b3 = np.zeros(512, dtype='f')
  w4 = np.float32(ss.truncnorm.rvs(0, 0.04, size=[512, 512]))
  b4 = np.zeros(512, dtype='f')
  w5 = np.float32(ss.truncnorm.rvs(0, 1/512.0, size=[512, 10]))
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

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply([cross_entropy_mean, total_loss])
  return loss_averages_op

TRAINING_ITERATION = 1000
RESTRUCT_ITERATION = 10000
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1
BATCH_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = parameters_init()
last_lr = INITIAL_LEARNING_RATE
_iter = 0
while _iter < TRAINING_ITERATION:
  parameters = parameters_conf(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)
  global_step = tf.Variable(_iter, trainable=False)
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCK_FOR_TRAIN / BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  learning_rate = tf.train.exponential_decay(learning_rate=last_lr, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
  images, labels = cifar10.distorted_inputs()
  dropout_prob = 0.2
  logits = inference(x, parameters, dropout_prob)
  cost = loss(logits, labels)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  init = tf.global_variables_initializer()
  sub_iter = 0
  with tf.Session(tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    aver_cost = 0.0
    start = time.time()
    while sub_iter < RESTRUCT_ITERATION:
      # Train
      _, c = sess.run([optimizer, cost])
      aver_cost += c / BATCH_SIZE
      sub_iter ++
      if sub_iter % 10 == 0:
        duration = time.time() - start
        print("iter %d: loss %f time %f" % (sub_iter, aver_cost, duration))
        aver_cost = 0.0
        start = time.time()
  _iter += sub_iter
