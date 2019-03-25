import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)[:, np.newaxis]
noise = np.random.normal(0, 1, size=x.shape)
y = np.power(x, 2) + noise

tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.float32, y.shape)

l1 = tf.layers.dense(inputs = tf_x, units = 20, activation = tf.nn.tanh, 
	kernel_initializer = tf.initializers.random_normal(mean=0, stddev=1))


l2 = tf.layers.dense(inputs = l1, units = 20, activation = tf.nn.relu, 
	kernel_initializer = tf.initializers.random_normal(mean=0, stddev=1))
output = tf.layers.dense(l2, 1)

loss = tf.losses.mean_squared_error(tf_y, output)
# loss = tf.reduce_mean(tf.square(tf_y-output))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(900):
	sess.run(optimizer, feed_dict={tf_x: x, tf_y: y})
	if(step % 10 == 0):
		print("step ", step, " loss:", sess.run(loss, feed_dict={tf_x: x, tf_y: y}))

predict = sess.run(output, feed_dict={tf_x: x})
plt.figure()
plt.scatter(x, y)
plt.plot(x, predict)
plt.show()