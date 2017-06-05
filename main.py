from common.ImageSegmentor import Segmentor
from common.ImageTransparator import ImageTransparater
from common.GradientImageSegmentor import ImageSegmentor
from traning.neural_network import NeuralNetwork
from common.DataCollection import DataCollection, transform_image_path_to_one_dimensional_matrices

TRAINING_DATA_PATH = "data_training/extracted_images"
IMAGE_FILE_NAME = "/Users/edwardsujono/Python_Project/math_image_recognition/api/e1cfe463-751e-4cc5-90ee-fc5dd236e6ab0.png"

if __name__ == "__main__":
	# first segmenting all the image first
	# file_name_path = "img/test2.jpg"

	# make the image transparent
	# image_transparantor = ImageTransparater()
	# image_transparantor.start_transforming(IMAGE_FILE_NAME)

	segmentor = Segmentor()
	segmentor.start_segmenting_image(file_image_part=IMAGE_FILE_NAME)
	# image_segmentor = ImageSegmentor("out.png")
	# image_segmentor.start_dsu_image_segmenting()

	# data_collection = DataCollection(image_directory=None)
	# data_collection.init_validation_data()
	# # num_data = data_collection.init_train_data()
	# data_collection.init_test_data()

	#
	# # test simple 2 percetron layer
	#
	# neural_network = NeuralNetwork()
	# for i in range(2):
    #
	# 	# neural_network.start_training(num_data=num_data, data_collection }=data_collection)
	# 	image_path = "/Users/edwardsujono/Python_Project/math_image_recognition/api/d169fa61-3220-4f35-b02c-8450c53f293c0.png"
	# 	data_test = transform_image_path_to_one_dimensional_matrices(image_path)
	# 	neural_network.restore_the_model([data_test] , is_data_collection=False)
