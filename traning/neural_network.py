import tensorflow as tf


class NeuralNetwork:

    number_node_1 = 500
    number_node_2 = 500
    number_node_3 = 500

    def __init__(self):
        # basically the ide
        self.hidden_layer_1 = {
            "weight": tf.Variable(tf.RandomNormal([784, self.number_node_1]))
        }