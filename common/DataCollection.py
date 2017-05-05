import os
import numpy as np
from skimage import io


def transform_image_path_to_one_dimensional_matrices(image_path):
    image_in_pixel = np.resize(io.imread(image_path), (1, 45*45))[0]
    return image_in_pixel


class DataCollection:

    def __init__(self, image_directory):
        list_directory = os.listdir(image_directory)
        self.data_directory = {}
        self.data_train_x = []
        self.data_train_y = []
        self.data_test_x = []
        self.data_test_y = []
        for label in list_directory:
            image_directory_path = image_directory + "/" + label
            for image_path in image_directory_path:
                self.data_directory[label] = image_directory_path + "/" + image_path

        for key,value in self.data_directory.iteritems():
            # 80% is for training
            # need to resize the image into 1 dimensional image
            directory_size = int(len(value) * 0.8)
            for i in range(directory_size):
                single_image_path = value[i]
                image_in_pixel = transform_image_path_to_one_dimensional_matrices(single_image_path)
                self.data_train_x.append(image_in_pixel)
                self.data_train_y.append(key)
            for i in range(directory_size, len(value)):
                self.data_test_x.append(transform_image_path_to_one_dimensional_matrices(value[i]))
                self.data_test_y.append(key)

        self.start_index = 0

        return

    def train_next_batch(self, batch_size):
        last_index = self.start_index + batch_size
        list_data_return = (self.data_train_x[self.start_index:last_index],
                            self.data_train_y[self.start_index:last_index])
        self.start_index = last_index
        return list_data_return


