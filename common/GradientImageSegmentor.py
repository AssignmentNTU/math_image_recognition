import numpy as np
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage import io
from PIL import Image


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

        for x in range(self.x_range):

            list_added = []

            for y in range(self.y_range):
                list_added.append(index)
                list_save_on_dict = [(y, x)]
                self.dictionary_location[index] = list_save_on_dict
                index += 1

            self.parent_list.append(list_added)

        return

    # DSU

    def start_image_segmenting(self, x_start, y_start, index):
        self.visited[y_start][x_start] = 1

        if self.parent_list[y_start][x_start] == 1:
            self.parent_list[y_start][x_start] = index
            self.dictionary_location[index].append((y_start, x_start))

        for x_change in (0, -1, 1):
            for y_change in (0, -1, 1):
                x_end = x_start+x_change
                y_end = y_start+y_change
                if self.is_save(x_end, y_end, self.x_range, self.y_range):
                    return self.start_image_segmenting(x_end, y_end, self.parent_list[y_start][x_start])

    def is_save(self, x, y , max_x, max_y):

        if x >= max_x or y >= max_y or x < 0 or  y < 0 or self.visited[x][y] == 1:
            return False

        return True

    def start_collecting_image_segmentation(self):
        number_part = 0 
        for key, value in self.dictionary_location.iteritems():
            if len(value) > self.LIMIT_PIXEL:
                list_x = []
                list_y = []

                for pixel_location in value:
                    list_x.append(pixel_location[0])
                    list_y.append(pixel_location[1])

                min_x = min(list_x)
                min_y = min(list_y)

                max_x = max(list_x)
                max_y = max(list_y)

                image_part = self.image_pil.crop((min_x, min_y, max_x, max_y))

        return
