from  traning.neural_network import NeuralNetwork
from DataCollection import transform_image_path_to_one_dimensional_matrices, resize_image_array


class ImageTranslator:

	def __init__(self):
		self.neural_network = NeuralNetwork()
		return

	def translate_image_to_string(self, image):

		image_array = transform_image_path_to_one_dimensional_matrices(image)

		result = self.neural_network.restore_the_model(data=[image_array], is_data_collection=False)
		return result

	def translate_numpyarray_to_string(self, image_array):

		image_array = resize_image_array(image_array)

		result = self.neural_network.restore_the_model(data=[image_array], is_data_collection=False)
		return result
