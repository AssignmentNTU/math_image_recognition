import numpy as np
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage import io
from PIL import Image
import Queue as Q


class ImageSegmentor:

    LIMIT_PIXEL = 10

    def __init__(self, image_name):
        self.image = io.imread(image_name)
        self.image_pil = Image.open(image_name)
        self.image_boundaries = find_boundaries(self.image, mode='router').astype(np.uint8)

        self.x_range = self.image.shape[1]
        self.y_range = self.image.shape[0]

        self.visited = np.zeros((self.y_range, self.x_range))
        self.parent_list = []

        self.dictionary_location = {}

        index = 0

        for x in range(self.y_range):

            list_added = []

            for y in range(self.x_range):
                list_added.append(index)
                # min_y, min_x, max_x, max_y
                list_save_on_dict = [10000, 10000, 0, 0]
                self.dictionary_location[index] = list_save_on_dict
                index += 1

            self.parent_list.append(list_added)

        return

    def start_dsu_image_segmenting(self):
        stack = Q.PriorityQueue()
        stack.put((self.image_boundaries[0][0], (0, 0, 0)))
        self.start_image_segmenting(stack)
        self.start_collecting_image_segmentation()

    # DSU

    def start_image_segmenting(self, stack):

        # cannot use recursion function

        while stack.not_empty:
            if stack.qsize() == 0:
                return
            pixel_location = stack.get()
            pixel_data = pixel_location[1]
            y_start = pixel_data[0]
            x_start = pixel_data[1]
            index = pixel_data[2]
            self.visited[y_start][x_start] = 1

            if 1 in self.image_boundaries[y_start][x_start]:
                self.parent_list[y_start][x_start] = index
                # record the max and the min
                data_record = self.dictionary_location[index]
                min_x = data_record[0]
                min_y = data_record[1]
                max_x = data_record[2]
                max_y = data_record[3]
                if y_start < min_y:
                    min_y = y_start

                if y_start > max_y:
                    max_y = y_start

                if x_start < min_x:
                    min_x = x_start

                if x_start > max_x:
                    max_x = x_start

                data_save = [min_x, min_y, max_x, max_y]
                self.dictionary_location[index] = data_save

            try:
                index = self.parent_list[y_start][x_start]
            except Exception as e:
                print("y_start: %d, x_start: %d" % (y_start, x_start))

            for x_change in (0, -1, 1):
                for y_change in (0, -1, 1):
                    x_end = x_start+x_change
                    y_end = y_start+y_change
                    if self.is_save(x_end, y_end, self.x_range, self.y_range):
                        x_start, y_start = x_end, y_end
                        self.visited[y_start][x_start] = 1
                        if 1 in self.image_boundaries[y_end][x_end]:
                            input_data = 1
                        else:
                            input_data = 0
                        stack.put((input_data* -1,  (y_end, x_end, index)))

    def is_save(self, x, y, max_x, max_y):

        if x >= max_x or y >= max_y or x < 0 or y < 0 or self.visited[y][x] == 1:
            return False

        return True

    def check_visited_graph(self):
        for i in range(len(self.y_range)):
            for j in range(len(self.x_range)):
                if (self.visited[j][i]) == 0:
                    return True
        return False

    # complexity is to high using this method
    # m x n x m x n

    def start_collecting_image_segmentation(self):
        number_part = 0
        for key, value in self.dictionary_location.iteritems():

            min_x = value[0]
            min_y = value[1]

            max_x = value[2]
            max_y = value[3]

            if max_y-min_y > self.LIMIT_PIXEL or max_x-min_x > self.LIMIT_PIXEL:
                image_part = self.image_pil.crop((min_x, min_y, max_x, max_y))
                image_part.save("image_part_"+str(number_part)+".png")
                number_part += 1
