import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 50
n_batch = mnist.train.num_examples // batch_size

tf_x = tf.placeholder(tf.float32, [None, 784])
tf_y = tf.placeholder(tf.float32, [None, 10])
tf_is_train = tf.placeholder(tf.bool, None)

l1 = tf.layers.dense(inputs = tf_x, units = 512, activation = tf.nn.relu, 
	kernel_initializer = tf.initializers.random_normal(mean=0, stddev=1))
# l12 = tf.layers.batch_normalization(l1, training=tf_is_train)
l1d = tf.layers.dropout(inputs = l1, rate=0.1, training=tf_is_train)


l2 = tf.layers.dense(inputs = l1d, units = 256, activation = tf.nn.relu, 
	kernel_initializer = tf.initializers.random_normal(mean=0, stddev=1))
# l22 = tf.layers.batch_normalization(l2, training=tf_is_train)
l2d = tf.layers.dropout(inputs = l2, rate=0.1, training=tf_is_train)

output = tf.layers.dense(l2d, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output) 
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lables=tf_y, logits=output))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

accuracy = tf.metrics.accuracy(
	labels=tf.argmax(tf_y, 1), predictions=tf.argmax(output, 1))[1]

corr = tf.equal(tf.argmax(output, 1), tf.argmax(tf_y,1))
accuracy1 = tf.reduce_mean(tf.cast(corr, tf.float32))


with tf.Session() as sess:
	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
	for epoch in range(50):
		for batch in range(n_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			sess.run(optimizer, feed_dict={tf_x: batch_x, tf_y: batch_y, tf_is_train: True})
		acc=sess.run(accuracy, feed_dict={tf_x: mnist.test.images, tf_y: mnist.test.labels, tf_is_train:False})
		acc1=sess.run(accuracy1, feed_dict={tf_x: mnist.test.images, tf_y: mnist.test.labels, tf_is_train:False})
		print("epoch ", epoch, " accuracy:", acc," accuracy1:", acc1)
