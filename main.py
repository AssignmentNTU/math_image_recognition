from common.ImageSegmentor import Segmentor
from common.ImageTransparator import ImageTransparater
from common.GradientImageSegmentor import ImageSegmentor
from traning.neural_network import NeuralNetwork
from common.DataCollection import DataCollection, transform_image_path_to_one_dimensional_matrices
import pickle

TRAINING_DATA_PATH = "img/extracted_images"
IMAGE_FILE_NAME = "/home/edwardsujono/Desktop/MachineLearning/cnn/img/device_crop.jpg"

if __name__ == "__main__":
	# first segmenting all the image first
	# file_name_path = "img/test2.jpg"

	# make the image transparent
	image_transparantor = ImageTransparater()
	trans_image = image_transparantor.start_transforming(IMAGE_FILE_NAME)

	segmentor = Segmentor()
	list_result = segmentor.start_segmenting_image(file_image_part=trans_image, pass_image_name=True, file_non_transparent=IMAGE_FILE_NAME)
	# image_segmentor = ImageSegmentor("out.png")
	# image_segmentor.start_dsu_image_segmenting()

	# data_collection = DataCollection(image_directory=TRAINING_DATA_PATH)
	# data_collection.init_validation_data()
	# # num_data = data_collection.init_train_data()
	# data_collection.init_test_data()

	#
	# # test simple 2 percetron layer
	#
	# neural_network = NeuralNetwork()
	# # for i in range(2):
	# #
	# # neural_network.start_training(num_data=num_data, data_collection =data_collection)
	# image_path = "/home/edwardsujono/Desktop/MachineLearning/cnn/img/8fa242ed-2314-48b5-8bff-91345b51f6ce.jpg"
	# data_test = transform_image_path_to_one_dimensional_matrices(IMAGE_FILE_NAME)
	# result = neural_network.restore_the_model([data_test], is_data_collection=False)
	with open("key_args.pickle", "rb") as handle:
		save_args = pickle.load(handle)
	for result in list_result:
		print save_args[result]
