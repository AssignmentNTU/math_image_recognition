import tensorflow as tf
# using tensorflow data example
from tensorflow.examples.tutorials.mnist import input_data

class NeuralNetwork:

	number_node_1 = 500
	number_node_2 = 500
	number_node_3 = 500
	output_classes = 10
	batch_size = 100

	def __init__(self):
		# basically the ide
		self.hidden_layer_1 = {
			"weight": tf.Variable(tf.random_normal([784, self.number_node_1])),
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

		self.x = tf.placeholder('float', [None, 784])
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

	def start_training(self):
		prediction = self.neural_network_model()
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)
		mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
		hm_epoc = 10
		# how many time we want to reloop the training
		with tf.Session() as sess:

			sess.run(tf.initialize_all_variables())

			for epoch in range(hm_epoc):
				loss_epoch = 0
				for _ in range(int(mnist.train.num_examples/self.batch_size)):
					epoch_x, epoch_y = mnist.train.next_batch(self.batch_size)
					_, c = sess.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
					loss_epoch += c
				correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
				print ("Epoch: %d Loss Epoch: %d, Acurracy: %f" % (epoch, loss_epoch,  accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels})))
