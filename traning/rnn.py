import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
# using tensorflow data example
from tensorflow.examples.tutorials.mnist import input_data


class NeuralNetwork:

	# 82 classification
	output_classes = 82
	batch_size = 128
	chunk_size = 45
	n_chunk = 45
	rnn_size = 128

	def __init__(self):
		# basically the ide

		self.output_layer = {
			"weight": tf.Variable(tf.random_normal([self.rnn_size, self.output_classes])),
			"biases": tf.Variable(tf.random_normal([self.output_classes]))
		}

		self.x = tf.placeholder('float', [None, self.n_chunk, self.chunk_size])
		self.y = tf.placeholder('float')

	def neural_network_model(self):
		data = self.x
		# activation function essentially just  x = 0 if x < 0 else x
		data = tf.unstack(data, self.n_chunk, 1)
		lstm_cell = rnn.BasicLSTMCell(self.rnn_size)
		output, states = rnn.static_rnn(lstm_cell, data,  dtype=tf.float32)
		output_layer = tf.add(tf.matmul(output[-1], self.output_layer["weight"]), self.output_layer["biases"])

		return output_layer

	def start_training(self, num_data, data_collection=None):
		prediction = self.neural_network_model()
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		if data_collection is None:
			mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
		else:
			mnist_own = data_collection

		hm_epoc = 50

		# how many time we want to re-loop the training

		with tf.Session() as sess:

			sess.run(tf.initialize_all_variables())

			for epoch in range(hm_epoc):
				loss_epoch = 0

				if data_collection is None:
					limit_range = mnist.train.num_examples
				else:
					limit_range = num_data

				for _ in range(int(limit_range/self.batch_size)):
					if data_collection is None:
						epoch_x, epoch_y = mnist.train.next_batch(self.batch_size)
					else:
						epoch_x, epoch_y = mnist_own.train_next_batch(self.batch_size)
						epoch_x = np.array(epoch_x).reshape(self.batch_size, self.n_chunk, self.chunk_size)
					_, c = sess.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})

					loss_epoch += c

				correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
				data_collection.restart_the_start_index()

				if data_collection is None:
					print ("Epoch: %d Loss Epoch: %d, Acurracy: %f" % (epoch, loss_epoch,  accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels})))
				else:
					list_data_train_x = mnist_own.get_test_data()[0]
					print (
						"Epoch: %d Loss Epoch: %d, Acurracy: %f" %
						(epoch, loss_epoch, accuracy.eval({self.x: np.asarray(list_data_train_x).reshape(len(list_data_train_x), self.n_chunk, self.chunk_size), self.y: mnist_own.get_test_data()[1]})))
