from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import scipy.stats as stats
import numpy as np
import time

import decomposition
# Parameters
learning_rate = 0.001
training_epochs = 130
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.dropout(layer_1, 0.8)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.dropout(layer_2, 0.8)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# init weights
w1 = np.random.randn(n_input, n_hidden_1)
w2 = np.random.randn(n_hidden_1, n_hidden_2)
w3 = np.random.randn(n_hidden_2, n_classes)
b1 = np.random.randn(n_hidden_1)
b2 = np.random.randn(n_hidden_2)
b3 = np.random.randn(n_classes)


epoch = 0
restructure_epoch = 5
rate = 0.05
stable = False

while epoch < training_epochs:

    # update weights
    weights = {
        'h1': tf.Variable(w1, dtype=tf.float32),
        'h2': tf.Variable(w2, dtype=tf.float32),
        'out': tf.Variable(w3, dtype=tf.float32)
    }
    biases = {
        'b1': tf.Variable(b1, dtype=tf.float32),
        'b2': tf.Variable(b2, dtype=tf.float32),
        'out': tf.Variable(b3, dtype=tf.float32)
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    # Launch the graph
    sub_epoch = 0
    with tf.Session() as sess:
        sess.run(init)
        if epoch > 0:
            #Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            #Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Post-restruction Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            # exit(0)
        while sub_epoch < restructure_epoch:
            start_time = time.time()
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # print(iv1.shape)
                # print(iv2.shape)
                # print(v1.shape)
                # print(v2.shape)

                # Compute average loss
                avg_cost += c / total_batch

            end_time = time.time()
            print("epoch training time: %s" % (end_time - start_time))
            # Display logs per epoch step
            print("Epoch:", '%04d' % (sub_epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))

            #Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            #Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            
            sub_epoch = sub_epoch + 1

        # print(v1)
        # print(v2)
        # max_v1 = np.max(v1)
        # max_v2 = np.max(v2)
        # print(v1[np.where(v1 > 0.05*max_v1)].shape)
        # print(v2[np.where(v2 > 0.05*max_v2)].shape)
        
        # exit(0)

        # restruction after each sub_epoch
        epoch += sub_epoch
        if not stable:
            start_time = time.time()
            w1 = weights['h1'].eval()
            w2 = weights['h2'].eval()
            w3 = weights['out'].eval()
            b1 = biases['b1'].eval()
            b2 = biases['b2'].eval()
            b3 = biases['out'].eval()

            # selected_1, selected_2 = decomposition.importance_dropout(v1, v2, rate=rate)
            # rate = rate*0.85
            # print("layers: %d %d" % (n_hidden_1, n_hidden_2))
            # n_hidden_1 = len(selected_1)
            # n_hidden_2 = len(selected_2)
            # print("reduced layers: %d %d" % (n_hidden_1, n_hidden_2))
            # w1 = w1[:, selected_1]
            # b1 = b1[selected_1]
            # w2 = w2[selected_1, :]
            # w2 = w2[:, selected_2]
            # b2 = b2[selected_2]
            # w3 = w3[selected_2, :]


            if n_hidden_1 == n_hidden_2:
                print("reduce rank of fc layer 1")
                print(w2.shape)

                ########## debug
                # s, v, d = np.linalg.svd(w2, full_matrices=False)
                # simp_v = v[np.where(v>v[0]*0.1)]
                # r = len(simp_v)
                # print("size of r: %s" % r)
                # simp_s = s[:, 0:r]
                # simp_d = d[0:r, :]
                # w2 = np.matmul(np.matmul(simp_s, np.diag(simp_v)), simp_d)
                # w2_temp = w2
                # flag = True
                ########## debug

                # flag, w1_update_factor, w2, r = mf_by_sgd(w2, n_hidden_2, n_hidden_1)
                # flag, w1_update_factor, w2, r = mf_by_svd(w2, n_hidden_2, n_hidden_1)
                # flag, w1_update_factor, w2, r = mf_by_half_identity(w2, n_hidden_2, n_hidden_1)

                ########## debug
                # w2_app = np.matmul(w1_update_factor, w2)
                # diff_abs = np.abs(w2_app - w2_temp)
                # print("max abs df: %s" % np.max(diff_abs))
                # print("mean abs df: %s" % np.mean(diff_abs))
                ########## debug
                flag, selected, r = decomposition.random_dropout(w2, n_hidden_2, n_hidden_1)
                # flag, selected, r = dropout_last(w2, n_hidden_2, n_hidden_1)
                
                epoch = epoch + 1
                if flag == True:
                    stable = True
                    break
                n_hidden_1 = r 
                print("reduced rank: %s" % r)
                # w1 = np.matmul(w1, w1_update_factor)
                # b1 = np.matmul(b1, w1_update_factor)
                w1 = w1[:, selected]
                b1 = b1[selected]
                w2 = w2[selected, :]
            else:
                print("reduce fc layer 2")
                # flag, w2_update_factor, w3, r = mf_by_sgd(w3, n_hidden_2, n_hidden_1, 1)
                # flag, w2_update_factor, w3, r = mf_by_svd(w3, n_hidden_2, n_hidden_1, 1)
                # flag, w2_update_factor, w3, r = mf_by_half_identity(w3, n_hidden_2, n_hidden_1)
                flag, selected, r = decomposition.random_dropout(w2, n_hidden_2, n_hidden_1)
                # flag, selected, r = dropout_last(w2, n_hidden_2, n_hidden_1)

                n_hidden_2 = r 
                # w2 = np.matmul(w2, w2_update_factor)
                # b2 = np.matmul(b2, w2_update_factor)
                w2 = w2[:, selected]
                b2 = b2[selected]
                w3 = w3[selected, :]
                # restructure_epoch = restructure_epoch * 2

            end_time = time.time()
            print("reconstruct time in epoch %s: %s" % (epoch, end_time - start_time))

