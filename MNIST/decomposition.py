
import numpy as np

import tensorflow as tf

def mf_by_sgd(W_, m, n, r=0):
# matrix factorization: W = W1 * W2. W: m*n W1: m*r W2: r*n
	max_iter = 2000
	start_learning_rate = 0.1
	# learning_rate = 0.01
	s, v, d = np.linalg.svd(W_)

	# print v
	simp_v = v[np.where(v>v[0]*0.1)]
	if r == 0:
		r = len(simp_v)
		# print len(v)
		# print ("m: %s  n: %s  r: %s" % (m, n, r))
		min_sz = min(m, n)
		if r == min_sz:
			print("converged!")
			return (True, np.identity(r), W_, r)	
		# if r < min_sz*0.9:
		# 	r = int(min_sz * 0.9) 

	# Factorized matrices
	W = tf.constant(W_, dtype=tf.float32)
	W1 = tf.Variable(tf.truncated_normal([m, r], stddev=0.2, mean=0))
	W2 = tf.Variable(tf.truncated_normal([r, n], stddev=0.2, mean=0))

	result = tf.matmul(W1, W2)
	diff = tf.subtract(W, result)
	# aver_diff = tf.reduce_mean(tf.abs(diff))
	aver_diff = tf.reduce_max(tf.abs(diff))

	# No regularization
	# cost = total_diff
	# With regularization
	lamda = 0.1
	# cost = total_diff + lamda*(tf.norm(W1) + tf.norm(W2))
	cost = aver_diff + lamda*(tf.reduce_mean(tf.square(W1)) + tf.reduce_mean(tf.square(W2)))

	# Use an exponentially decaying learning rate.
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 50, 0.8, staircase=True)
	# Passing global_step to minimize() will increment it at each step so
	# that the learning rate will be decayed at the specified intervals.
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	# train_step = optimizer.minimize(cost, global_step=global_step)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

	# Initializing the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for i in range(max_iter):
			# sess.run(train_step)
			sess.run([optimizer, cost])
			#print tf.subtract(W, result).eval()

			#print np.sum(np.square(np.subtract(W_, np.matmul(W1.eval(), W2.eval()))))
			#print(learning_rate.eval())
			# print("%s: %s" % (i, cost.eval()))
			# if(aver_diff.eval() < 0.000001):
			# 	print("iteration %s" % i)
			# 	print(diff.eval())
			# 	break
			#print(diff.eval())


		output_W1 = W1.eval()
		output_W2 = W2.eval()
		real_diff = np.subtract(W_, np.matmul(output_W1, output_W2))
		print np.max(np.abs(real_diff))
		# print real_diff
	# print "final diff"
	# print np.subtract(W_, np.matmul(output_W1, output_W2))
	# end
	return (False, output_W1, output_W2, r)

# W_ = W1 * W2, W1 = [I;B] W2 = S_rVD, estimate BS_r for S_(n-r)
def mf_by_svd(W_, m, n):
	max_iter = 2000
	start_learning_rate = 0.1

	s, v, d = np.linalg.svd(W_, full_matrices=False)

	simp_v = v[np.where(v>v[0]*0.1)]
	r = len(simp_v)
	print("r: %s" % r)
	if r==m and r==n:
		print("converged!")
		return (True, np.identity(r), W_, r)
	# print m
	# print n
	# print(v.shape)
	# print(np.diag(v).shape)
	# print(d.shape)

	simp_s = s[0:r, :]
	output_W2 = np.matmul(np.matmul(simp_s, np.diag(v)), d)

	B = tf.Variable(tf.truncated_normal([m - r, r], stddev=0.2, mean=0))
	C = tf.constant(s[0:r, 0:r], dtype=tf.float32)
	D = tf.constant(s[0:r, r:m], dtype=tf.float32)
	C2 = tf.constant(s[r:m, 0:r], dtype=tf.float32)
	D2 = tf.constant(s[r:m, r:m], dtype=tf.float32)

	diff1 = tf.matmul(B, C) - C2
	diff2 = tf.matmul(B, D) - D2

	lamda = 0.1
	aver_diff = tf.reduce_mean(tf.square(diff1)) + tf.reduce_mean(tf.square(diff2)) 
	# aver_diff = tf.reduce_max(tf.abs(diff1)) + tf.reduce_max(tf.abs(diff2))
	cost = aver_diff + lamda * tf.reduce_mean(tf.square(B))

	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 50, 0.8, staircase=True)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

	# Initializing the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for i in range(max_iter):
			# sess.run(train_step)
			sess.run([optimizer, cost])
			#print tf.subtract(W, result).eval()

			#print np.sum(np.square(np.subtract(W_, np.matmul(W1.eval(), W2.eval()))))
			#print(learning_rate.eval())
			# print("%s: %s" % (i, cost.eval()))
			if(aver_diff.eval() < 0.01):
				break
			#print(diff.eval())
		# print "opt done"
		# print tf.reduce_max(tf.abs(diff1)).eval()
		# BC_diff = np.subtract(C2.eval(), np.matmul(B.eval(), C.eval()))
		# print BC_diff
		# print np.max(np.abs(BC_diff))
		# print tf.reduce_max(tf.abs(diff2)).eval()
		# BD_diff = np.subtract(D2.eval(), np.matmul(B.eval(), D.eval()))
		# print BD_diff
		# print np.max(np.abs(BD_diff))
		output_W1 = np.zeros([m, r])
		output_W1[0:r, :] = np.identity(r)
		output_W1[r:m, :] = B.eval()
		real_diff = np.subtract(W_, np.matmul(output_W1, output_W2))
		print np.max(np.abs(real_diff))
	# print "final diff"
	# print np.subtract(W_, np.matmul(output_W1, output_W2))
	# end
	return (False, output_W1, output_W2, r)

# W_ = W1 * W2, W1 = [I;B], estimate BW2 for W_(n-r)
def mf_by_half_identity(W_, m, n):
	max_iter = 2000
	start_learning_rate = 0.1

	s, v, d = np.linalg.svd(W_, full_matrices=False)

	simp_v = v[np.where(v>v[0]*0.1)]
	r = len(simp_v)
	print("r: %s" % r)
	if r==m and r==n:
		print("converged!")
		return (True, np.identity(r), W_, r)

	output_W2 = W_[0:r, :]
	W_rest = W_[r:m, :] 
	W2 = tf.constant(output_W2, dtype=tf.float32)
	B = tf.Variable(tf.truncated_normal([m - r, r], stddev=0.2, mean=0))

	diff = tf.matmul(B, W2) - W_rest
	lamda = 0.1
	aver_diff = tf.reduce_mean(tf.square(diff))
	cost = aver_diff + 100*tf.reduce_mean(tf.square(B))

	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 50, 0.8, staircase=True)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

	# Initializing the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for i in range(max_iter):
			# sess.run(train_step)
			sess.run([optimizer, cost])
			#print tf.subtract(W, result).eval()

			#print np.sum(np.square(np.subtract(W_, np.matmul(W1.eval(), W2.eval()))))
			#print(learning_rate.eval())
			# print("%s: %s" % (i, cost.eval()))
			if(aver_diff.eval() < 0.000001):
				break
			#print(diff.eval())

		output_W1 = np.zeros([m, r])
		output_W1[0:r, :] = np.identity(r)
		output_W1[r:m, :] = B.eval()
		real_diff = np.subtract(W_, np.matmul(output_W1, output_W2))
		print np.max(np.abs(real_diff))
	# print "final diff"
	# print np.subtract(W_, np.matmul(output_W1, output_W2))
	# end
	return (False, output_W1, output_W2, r)	

def random_dropout(W_, m, n, rate = 0.05):

	if m == n:
		s, v, d = np.linalg.svd(W_, full_matrices=False)
		threshold = np.mean(v[0:10]) * rate
		print("threshold: %f" % threshold)
		simp_v = v[np.where(v>threshold)]
		r = len(simp_v)
		print("r: %d" % r)
		if r==m:
			print("converged!")
			return (True, range(0, r), r)

		rest = np.sort(np.random.choice(m, r, replace=False))
		return (False, rest, r)
	else:
		r = n
		rest = np.sort(np.random.choice(m, r, replace=False))
		return (False, rest, r)

def importance_dropout(v1, v2, rate=0.05):
	# temp1 = np.sort(v1)
	# temp2 = np.sort(v2)
	# threshold1 = np.mean(temp1[-10:]) * rate
	# threshold2 = np.mean(temp2[-10:]) * rate
	threshold1 = np.max(v1) * rate
	threshold2 = np.max(v2) * rate
	selected_1 = np.where(v1 > threshold1)
	selected_2 = np.where(v2 > threshold2)
	return (selected_1[0], selected_2[0])


def dropout_last(W_, m, n, rate = 0.05):

	if m == n:
		s, v, d = np.linalg.svd(W_, full_matrices=False)
		threshold = np.mean(v[0:10]) * rate
		simp_v = v[np.where(v>threshold)]
		r = len(simp_v)
		print("threshold: %f" % threshold)
		print("r: %s" % r)
		if r==m:
			print("converged!")
			return (True, range(0, r), r)
		rest = range(0, r)
		return (False, rest, r)
	else:
		r = n
		rest = range(0, r)
		return (False, rest, r)

def two_layer_random_dropout(W_, m, n):

	s, v, d = np.linalg.svd(W_, full_matrices=False)
	simp_v = v[np.where(v>v[0]*0.1)]
	r = len(simp_v)
	print("r: %s" % r)
	if r==m:
		print("converged!")
		return (True, range(0, r), r)
	rest = np.sort(np.random.choice(m, r, replace=False))
	return (False, rest, r)

# W = np.random.randn(600, 600)
# W1, W2 = mf_by_sgd(W, 600, 600, 500)
# W1, W2 = mf_by_svd(W, 600, 600, 500)
# print W
# print W1
# print W2
# diff = np.mean(np.square(np.subtract(W, np.matmul(W1, W2))))
# print diff
