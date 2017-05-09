from common.ImageSegmentor import Segmentor
from common.GradientImageSegmentor import ImageSegmentor
from traning.cnn import NeuralNetwork
from common.DataCollection import DataCollection

TRAINING_DATA_PATH = "img/extracted_images"

if __name__ == "__main__":
	# first segmenting all the image first
	# file_name_path = "img/test2.jpg"
	# segmentor = Segmentor()
	# segmentor.start_segmenting_image(file_image_part=file_name_path)
	# image_segmentor = ImageSegmentor("img/test1.jpg")
	# image_segmentor.start_dsu_image_segmenting()

	data_collection = DataCollection(image_directory=None)
	num_data = data_collection.init_train_data()
	data_collection.init_test_data()

	# test simple 2 percetron layer

	neural_network = NeuralNetwork()
	neural_network.start_training(num_data=num_data, data_collection=data_collection)
