import tensorflow as tf
import numpy as np
# using tensorflow data example
from tensorflow.examples.tutorials.mnist import input_data


class NeuralNetwork:

	# 82 classification
	output_classes = 10
	batch_size = 1000
	image_width = 28
	image_height = 28
	input_class = 28*28

	def __init__(self):

		# 5x5 patch with 1 input channel and give 32 output channel or kernel
		self.convolutional_1 = {
			"weight": tf.Variable(tf.random_normal([5, 5, 1, 32])),
			"biases": tf.Variable(tf.random_normal([32]))
		}

		self.normal_ = {"weight": tf.Variable(tf.random_normal([5, 5, 32, 64])),
						"biases": tf.Variable(tf.random_normal([64]))}
		self.convolutional_2 = self.normal_

		# this is normal perceptron layer
		self.full_connected_1 = {
			"weight": tf.Variable(tf.random_normal([7*7*64, 1028])),
			"biases": tf.Variable(tf.random_normal([1028]))
		}

		self.output_layer = {
			"weight": tf.Variable(tf.random_normal([1028, self.output_classes])),
			"biases": tf.Variable(tf.random_normal([self.output_classes]))
		}

		self.x = tf.placeholder('float', [None, self.input_class])
		self.y = tf.placeholder('float')

	def conv2d(self, x, weight, biases, strides=1):
		x = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.add(x, biases)
		return tf.nn.relu(x)

	def max_pooling(self, x, k=2):
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

	def neural_network_model(self):
		data = self.x
		data = tf.reshape(data, shape=[-1, self.image_width, self.image_height, 1])
		# activation function essentially just  x = 0 if x < 0 else x
		conv1 = self.conv2d(x=data, weight=self.convolutional_1.get("weight"), biases=self.convolutional_1.get("biases"))
		conv1 = self.max_pooling(conv1)

		conv2 = self.conv2d(x=conv1, weight=self.convolutional_2.get("weight"), biases=self.convolutional_2.get("biases"))
		conv2 = self.max_pooling(conv2)

		data = tf.reshape(conv2, [-1, 7*7*64])
		fc1 = tf.add(tf.matmul(data, self.full_connected_1.get("weight")), self.full_connected_1.get("biases"))
		fc1 = tf.nn.relu(fc1)

		out = tf.add(tf.matmul(fc1, self.output_layer.get("weight")), self.output_layer.get("biases"))

		return out

	def start_training(self, num_data, data_collection=None):
		prediction = self.neural_network_model()
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

		if data_collection is None:
			mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
		else:
			mnist_own = data_collection

		hm_epoc = 100

		# how many time we want to re-loop the training

		with tf.Session() as sess:

			sess.run(tf.initialize_all_variables())

			for epoch in range(hm_epoc):
				loss_epoch = 0

				if data_collection is None:
					limit_range = mnist.train.num_examples
				else:
					limit_range = num_data

				for index in range(int(limit_range/self.batch_size)):
					if data_collection is None:
						epoch_x, epoch_y = mnist.train.next_batch(self.batch_size)
					else:
						epoch_x, epoch_y = mnist_own.train_next_batch(self.batch_size)
					_, c = sess.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
					loss_epoch += c
					print ("index: %d, loss_epoch: %d" % (index, loss_epoch))

				correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
				if data_collection is not None:
					data_collection.restart_the_start_index()

				if data_collection is None:
					print ("Epoch: %d Loss Epoch: %d, Acurracy: %f" % (epoch, loss_epoch,  accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels})))
				else:
					list_data_train_x = mnist_own.get_test_data()[0]
					print (
						"Epoch: %d Loss Epoch: %d, Acurracy: %f" %
						(epoch, loss_epoch, accuracy.eval({self.x: list_data_train_x, self.y: mnist_own.get_test_data()[1]})))
