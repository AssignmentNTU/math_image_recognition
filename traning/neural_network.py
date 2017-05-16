import tensorflow as tf
# using tensorflow data example
from tensorflow.examples.tutorials.mnist import input_data


class NeuralNetwork:

	number_node_1 = 500
	number_node_2 = 500
	number_node_3 = 500
	# 82 classification
	output_classes = 82
	batch_size = 100

	def __init__(self):
		# basically the ide

		self.hidden_layer_1 = {
			"weight": tf.Variable(tf.random_normal([28*28, self.number_node_1])),
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

		self.x = tf.placeholder('float', [None, 28*28])
		self.y = tf.placeholder('float')

	def neural_network_model(self):
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

	def start_training(self, num_data, data_collection=None):
		prediction = self.neural_network_model()
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)
		saver = tf.train.Saver()

		if data_collection is None:
			mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
		else:
			mnist_own = data_collection

		hm_epoc = 1000

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

					_, c = sess.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})

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
					print (
						"Loss: Epoch: %d Loss Epoch: %d, Acurracy: %f" %
						(epoch, loss_epoch, accuracy.eval({self.x: mnist_own.get_test_data()[0], self.y: mnist_own.get_test_data()[1]})))

		saver.save(sess, "save_model.dat")
