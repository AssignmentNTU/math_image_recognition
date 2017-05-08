import os
import numpy as np
from skimage import io
import pickle

# put general function that could be used by other model later


def transform_image_path_to_one_dimensional_matrices(image_path):
    image_in_pixel = np.resize(io.imread(image_path), (1, 45*45))[0]
    return image_in_pixel


def generate_one_hot_encoding(result_class, num_class):
    list_zero = np.zeros((num_class, ), dtype=np.int)
    list_zero[result_class] = 1
    return list_zero


class DataCollection:

    NUM_CLASS = 82

    def __init__(self, image_directory=None):

        if not image_directory:
            self.start_index = 0
            return

        list_directory = os.listdir(image_directory)
        self.data_directory = {}
        self.data_train_x = []
        self.data_train_y = []
        self.data_test_x = []
        self.data_test_y = []

        # this is used for .one hot encoding label
        # example particular input x give class 2 from 5 as the result
        # 2 will be represented as [0,0,1,0,0]

        self.label_one_hot_encoding = {}

        index_label = 0
        num_class = len(list_directory)

        for label in list_directory:
            image_directory_path = image_directory + "/" + label
            for image_path in os.listdir(image_directory_path):
                if self.data_directory.get(label) is None:
                    self.data_directory[label] = []
                self.data_directory[label].append(image_directory_path + "/" + image_path)
            self.label_one_hot_encoding[label] = generate_one_hot_encoding(index_label, num_class)
            index_label += 1

        # will break the dictionary of data to the list which will be trained and tested

        for key, value in self.data_directory.iteritems():
            # 80% is for training
            # need to resize the image into 1 dimensional image
            directory_size = int(len(value) * 0.8)
            one_hot_label = self.label_one_hot_encoding.get(key)

            for i in range(directory_size):
                single_image_path = value[i]
                image_in_pixel = transform_image_path_to_one_dimensional_matrices(single_image_path)
                if len(image_in_pixel) != 2025:
                    continue
                self.data_train_x.append(image_in_pixel)
                self.data_train_y.append(one_hot_label)

            for i in range(directory_size, len(value)):
                self.data_test_x.append(transform_image_path_to_one_dimensional_matrices(value[i]))
                self.data_test_y.append(one_hot_label)

            print ("Finish Collecting data for key %s size %d" % (key, len(value)))

        self.start_index = 0

        # save the data into pickle

        with open("data_train_x.pickle", "wb") as handle:
            pickle.dump(self.data_train_x, handle)

        with open("data_train_y.pickle", "wb") as handle:
            pickle.dump(self.data_train_y, handle)

        with open("data_test_x.pickle", "wb") as handle:
            pickle.dump(self.data_test_x, handle)

        with open("data_test_y.pickle", "wb") as handle:
            pickle.dump(self.data_test_y, handle)

        return

    def init_train_data(self):

        with open("data_train_x.pickle", "rb") as handle:
            self.data_train_x = pickle.load(handle)

        with open("data_train_y.pickle", "rb") as handle:
            self.data_train_y = pickle.load(handle)

        # will return the number of train data

        return len(self.data_train_x)

    def init_test_data(self):

        with open("data_test_x.pickle", "rb") as handle:
            self.data_test_x = pickle.load(handle)

        with open("data_test_y.pickle", "rb") as handle:
            self.data_test_y = pickle.load(handle)

    def train_next_batch(self, batch_size):

        last_index = self.start_index + batch_size
        list_data_return = (self.data_train_x[self.start_index:last_index],
                            self.data_train_y[self.start_index:last_index])
        self.start_index = last_index
        return list_data_return

    def restart_the_start_index(self):
        self.start_index = 0

    def get_test_data(self):
        return self.data_test_x, self.data_test_y