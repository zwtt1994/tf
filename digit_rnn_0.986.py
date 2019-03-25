import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
time_step = 28          # rnn time step / image height
input_size = 28         # rnn input size / image width
n_batch = mnist.train.num_examples // batch_size

tf_x = tf.placeholder(tf.float32, [None, 784])
in_x = tf.reshape(tf_x, [-1, time_step, input_size]) 
tf_y = tf.placeholder(tf.float32, [None, 10])
tf_is_train = tf.placeholder(tf.bool, None)


# single RNN  learning_rate=1e-3
# rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=28)
# outputs1, (h_c, h_n) = tf.nn.dynamic_rnn(
#     cell=rnn_cell, inputs=in_x, initial_state=None, 
#     dtype=tf.float32, time_major=False
# )                                                # [bacth,28,64]

# double RNN learning_rate=1e-3
# num_units = [64, 64]
# cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in num_units]
# stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

# outputs, (h_c2, h_n2) = tf.nn.dynamic_rnn(
#     cell=stacked_rnn_cell, inputs=in_x, initial_state=None, 
#     dtype=tf.float32, time_major=False
# )       

# bidirectional RNN learning_rate=1e-3
fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)

# fw_initial_state = fw_cell.zero_state(64, tf.float32)
# bw_initial_state = bw_cell.zero_state(64, tf.float32)


outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
	inputs=in_x, cell_fw=fw_cell, cell_bw=fw_cell, time_major=False,
	initial_state_fw=None, initial_state_bw=None, dtype=tf.float32)
state = tf.concat([outputs[0][:,-1,:], outputs[1][:,-1,:]], 1)
# [bacth,28,64]


output = tf.layers.dense(state, 10)  # [bacth,10]  

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output) 
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lables=tf_y, logits=output))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

corr = tf.equal(tf.argmax(output, 1), tf.argmax(tf_y,1))
accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))


with tf.Session() as sess:
	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
	for epoch in range(20):
		for batch in range(n_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			sess.run(optimizer, feed_dict={tf_x: batch_x, tf_y: batch_y, tf_is_train: True})
		acc=sess.run(accuracy, feed_dict={tf_x: mnist.test.images, tf_y: mnist.test.labels, tf_is_train:False})
		# acc=sess.run(accuracy, feed_dict={tf_x: batch_x, tf_y: batch_y, tf_is_train:False})
		print("epoch ", epoch, " accuracy:", acc)
