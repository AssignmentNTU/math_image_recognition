import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class LinearRegression:

	def __init__(self, list_x, list_y):
		self.train_x = np.asarray(list_x)
		self.train_y = np.asarray(list_y)
		self.learning_rate = 0.01
		self.display_step = 50
		self.training_epoch = 1000

	def start_training(self):
		rng = np.random

		X = tf.placeholder("float")
		Y = tf.placeholder("float")

		n_sample = self.train_x.shape[0]

		# the equation would be Y = W(X) + B
		W = tf.Variable(rng.rand(), name="weight")
		B = tf.Variable(rng.rand(), name="weight")

		# linear model
		prediction = tf.add(tf.matmul(X,W), B)

		cost = tf.reduce_sum(tf.pow(prediction-Y, 2))/(2*n_sample)

		# use Gradient Descent
		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

		with tf.Session() as sess:

			for epoch in range(self.training_epoch):
				for (x, y) in zip(self.train_x, self.train_y):
					sess.run(optimizer, feed_dict={X: x, Y: y})