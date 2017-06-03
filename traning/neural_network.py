import tensorflow as tf
# using tensorflow data example
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:

	number_node_1 = 500
	number_node_2 = 500
	number_node_3 = 500
	# 82 classification
	output_classes = 82
	batch_size = 100
	pixel_size = 45

	def __init__(self):
		# basically the ide
		self.list_accuracy = []
		self.list_time = []
		self.axis = [0, 200, 0, 1]

		self.hidden_layer_1 = {
			"weight": tf.Variable(tf.random_normal([self.pixel_size*self.pixel_size, self.number_node_1])),
			"biases": tf.Variable(tf.random_normal([self.number_node_1]))
		}

		self.hidden_layer_2 = {
			"weight": tf.Variable(tf.random_normal([self.number_node_1, self.number_node_2])),
			"biases": tf.Variable(tf.random_normal([self.number_node_2]))
		}

		self.hidden_layer_3 = {
			"weight": tf.Variable(tf.random_normal([self.number_node_2, self.number_node_3])),
			"biases": tf.Variable(tf.random_normal([self.number_node_3]))
		}

		self.output_layer = {
			"weight": tf.Variable(tf.random_normal([self.number_node_3, self.output_classes])),
			"biases": tf.Variable(tf.random_normal([self.output_classes]))
		}

		self.x = tf.placeholder('float', [None, self.pixel_size*self.pixel_size])
		self.y = tf.placeholder('float')

		self.real_input_x = tf.placeholder('float', [self.pixel_size*self.pixel_size])

	def neural_network_model(self, is_real=False):
		data = self.x
		# activation function essentially just  x = 0 if x < 0 else x
		layer_1 = tf.add(tf.matmul(data, self.hidden_layer_1["weight"]), self.hidden_layer_1["biases"])
		layer_1 = tf.nn.relu(layer_1)

		layer_2 = tf.add(tf.matmul(layer_1, self.hidden_layer_2["weight"]), self.hidden_layer_2["biases"])
		layer_2 = tf.nn.relu(layer_2)

		layer_3 = tf.add(tf.matmul(layer_2, self.hidden_layer_3["weight"]), self.hidden_layer_3["biases"])
		layer_3 = tf.nn.relu(layer_3)

		output_layer = tf.add(tf.matmul(layer_3, self.output_layer["weight"]), self.output_layer["biases"])

		return output_layer

	def start_training(self, num_data, data_collection=None, from_checkpoint=False):
		prediction = self.neural_network_model()
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
		optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)
		saver = tf.train.Saver()

		if data_collection is None:
			mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
		else:
			mnist_own = data_collection

		hm_epoc = 3000

		# how many time we want to re-loop the training

		with tf.Session() as sess:

			sess.run(tf.initialize_all_variables())

			if from_checkpoint:
				saver = tf.train.import_meta_graph('deep_learning_model-2000.meta')
				saver.restore(sess, tf.train.latest_checkpoint('./'))

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
					# print("index count: %d, loss_epoch: %d" % (index, loss_epoch))
					loss_epoch += c

				correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
				if data_collection is not None:
					data_collection.restart_the_start_index()

				# for validation just for own mnist data
				if data_collection is not None:
					print (
						"Validation: Epoch: %d Loss Epoch: %d, Acurracy: %f" %
						(epoch, loss_epoch,
							accuracy.eval({self.x: mnist_own.get_validation_data()[0], self.y: mnist_own.get_validation_data()[1]})))

				if data_collection is None:
					print ("Epoch: %d Loss Epoch: %d, Acurracy: %f" % (epoch, loss_epoch,  accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels})))
				else:
					accuracy_value = accuracy.eval({self.x: mnist_own.get_test_data()[0], self.y: mnist_own.get_test_data()[1]})
					print (
						"Loss: Epoch: %d Loss Epoch: %d, Acurracy: %f" %
						(epoch, loss_epoch, accuracy_value))
					self.list_time.append(epoch)
					self.list_accuracy.append(accuracy_value)
		plt.plot(self.list_time,self.list_accuracy)
		plt.axis(self.axis)
		plt.show()
		saver.save(sess, "deep_learning_model", global_step=10000)

	def restore_the_model(self, data_collection):
		data = None
		if data_collection is not None:
			data = [data_collection.get_validation_data()[0][0]]
		# self.print_sample_data(data)
		prediction = self.neural_network_model(is_real=False)
		saver = tf.train.Saver()
		with tf.Session() as session:

			# saver = tf.train.import_meta_graph('deep_learning_model-2000.meta')
			saver.restore(session, tf.train.latest_checkpoint('./'))
			# session.run(tf.global_variables_initializer())
			result = session.run(prediction, feed_dict={self.x: data})

		result = np.argmax(result, axis=1)
		print ("result: %d" % result)
		return result

	def print_sample_data(self, data_array):
		data_array = data_array[0]
		for data in data_array:
			print (data)

		print("\n")
