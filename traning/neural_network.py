import tensorflow as tf


class NeuralNetwork:

    number_node_1 = 500
    number_node_2 = 500
    number_node_3 = 500
    output_classes = 10
    batch_size = 100

    def __init__(self):
        # basically the ide
        self.hidden_layer_1 = {
            "weight": tf.Variable(tf.RandomNormal([784, self.number_node_1])),
            "biases": tf.Variable(tf.RandomNormal([self.number_node_1]))
        }

        self.hidden_layer_2 = {
            "weight": tf.Variable(tf.RandomNormal([self.number_node_1, self.number_node_2])),
            "biases": tf.Variable(tf.RandomNormal([self.number_node_2]))
        }

        self.hidden_layer_3 = {
            "weight": tf.Variable(tf.RandomNormal([self.number_node_2, self.number_node_3])),
            "biases": tf.Variable(tf.RandomNormal([self.number_node_3]))
        }

        self.output_layer = {
            "weight": tf.Variable(tf.RandomNormal([self.number_node_3, self.output_classes])),
            "biases": tf.Variable(tf.RandomNormal([self.output_classes]))
        }

    def neural_network_model(self, data):
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
