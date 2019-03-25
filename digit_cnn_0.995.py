import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

tf_x = tf.placeholder(tf.float32, [None, 784])
in_x = tf.reshape(tf_x, [-1, 28, 28, 1]) 
tf_y = tf.placeholder(tf.float32, [None, 10])
tf_is_train = tf.placeholder(tf.bool, None)


conv_11 = tf.layers.conv2d( 
    inputs=in_x, filters=32, kernel_size=3,strides=1, padding='same', activation=tf.nn.relu
)   #  (28, 28, 1) -> (28, 28, 16)
# pool1 = tf.layers.max_pooling2d(
# 	inputs=conv_11, pool_size=2, strides=2,
# )           # -> (14, 14, 16)
conv_12 = tf.layers.conv2d( 
    inputs=conv_11, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu
)   #  (14, 14, 32) -> (14, 14, 32)
conv_13 = tf.layers.conv2d( 
    inputs=conv_12, filters=64, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu
)   #  (28, 28, 32) -> (14, 14, 64)
# pool2 = tf.layers.max_pooling2d(
# 	inputs=conv_12, pool_size=2, strides=2,
# )           # -> (7, 7, 32)
# conv_13 = tf.layers.conv2d( 
#     inputs=conv_12, filters=64, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu
# )   #  (28, 28, 32) -> (14, 14, 64)
# l22 = tf.layers.batch_normalization(l2, training=tf_is_train)
ld_14 = tf.layers.dropout(inputs = conv_13, rate=0.4, training=tf_is_train)

conv_21 = tf.layers.conv2d( 
    inputs=ld_14, filters=32, kernel_size=3,strides=1, padding='same', activation=tf.nn.relu
)   #  (14, 14, 64) -> (14, 14, 32)
conv_22 = tf.layers.conv2d( 
    inputs=conv_21, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu
)   #  (14, 14, 32) -> (14, 14, 32)
conv_23 = tf.layers.conv2d( 
    inputs=conv_22, filters=64, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu
)   #  (14, 14, 32) -> (7, 7, 64)
ld_24 = tf.layers.dropout(inputs = conv_23, rate=0.4, training=tf_is_train)

flat = tf.reshape(ld_24, [-1, 7*7*64])  
l31 = tf.layers.dense(flat, 128) 
ld_32 = tf.layers.dropout(inputs = l31, rate=0.4, training=tf_is_train)
output = tf.layers.dense(ld_32, 10) 

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
		print("epoch ", epoch, " accuracy:", acc)
