from common.ImageSegmentor import Segmentor
from common.ImageTransparator import ImageTransparater
from common.GradientImageSegmentor import ImageSegmentor
from traning.neural_network import NeuralNetwork
from common.DataCollection import DataCollection

TRAINING_DATA_PATH = "data_training/extracted_images"
IMAGE_FILE_NAME = "upload_image/b0f8204a-4c30-4d97-a351-86606d808186.png"

if __name__ == "__main__":
	# first segmenting all the image first
	# file_name_path = "img/test2.jpg"

	# make the image transparent
	# image_transparantor = ImageTransparater()
	# image_transparantor.start_transforming(IMAGE_FILE_NAME)

	# segmentor = Segmentor()
	# segmentor.start_segmenting_image(file_image_part=IMAGE_FILE_NAME)
	# image_segmentor = ImageSegmentor("out.png")
	# image_segmentor.start_dsu_image_segmenting()

	data_collection = DataCollection(image_directory=None)
	data_collection.init_validation_data()
	# num_data = data_collection.init_train_data()
	data_collection.init_test_data()

	#
	# # test simple 2 percetron layer
	#
	neural_network = NeuralNetwork()
	# neural_network.start_training(num_data=num_data, data_collection }=data_collection)
	neural_network.restore_the_model(data_collection)
