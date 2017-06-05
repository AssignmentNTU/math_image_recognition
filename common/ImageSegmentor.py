import numpy as np
import uuid
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage import io
from PIL import Image
from ImageTranslator import ImageTranslator
# This class will brute force the image contour and crop it


class Segmentor:

	# LIMIT NUMBER GAP MEANS THAT TOTAL COLUMN 0
	# IN THE IMAGE
	LIMIT_NUMBER_GAP = 1
	DIRECTORY_PATH = "/Users/edwardsujono/Python_Project/math_image_recognition/api/"

	def __init__(self):
		return

	def start_segmenting_image(self, file_image_part, pass_image_name=True):

		list_result = []
		image_translator = ImageTranslator()
		if pass_image_name:
			image = io.imread(file_image_part)
		else:
			image = file_image_part

		image_pil = Image.open(file_image_part)
		result_boundaries = find_boundaries(image, mode='outer').astype(np.uint8)

		x_range = image.shape[1]		# width
		y_range = image.shape[0]		# height

		list_x = []
		list_y = []

		number_gap = 0
		number_part = 0

		get_pixel_start = False

		for i in range(x_range):
			# gap means that it is different segment of image
			gap = False

			for j in range(y_range):
				# this is gray scale image
				if len(image.shape) == 2:
					if result_boundaries[j][i] == 1:
						list_y.append(j)
						list_x.append(i)
						gap = True
				# this is RGB value
				else:
					if self.check_black(image[j][i]):
						list_y.append(j)
						list_x.append(i)
						get_pixel_start = True
						gap = True

			if not gap and get_pixel_start:
				number_gap += 1

			if number_gap == self.LIMIT_NUMBER_GAP:

				number_gap = 0

				if len(list_x) == 0 or len(list_y) == 0:
					continue

				min_x = min(list_x)
				max_x = max(list_x)

				min_y = min(list_y)
				max_y = max(list_y)

				image_part = image_pil.crop((min_x, min_y, max_x, max_y))
				file_save_name = self.DIRECTORY_PATH + str(uuid.uuid4()) + str(number_part) + ".png"
				image_part.save(file_save_name)
				image_part.close()
				list_result.append(image_translator.translate_image_to_string(file_save_name))

				number_part += 1

				list_x = []
				list_y = []

			# last one usually there is no gap in between

		if len(list_x) == 0 or len(list_y) == 0:
			return list_result

		min_x = min(list_x)
		max_x = max(list_x)

		min_y = min(list_y)
		max_y = max(list_y)

		image_part = image_pil.crop((min_x, min_y, max_x, max_y))
		file_save_name = self.DIRECTORY_PATH + str(uuid.uuid4()) + str(number_part) + ".png"
		image_part.save(file_save_name)
		image_part.close()
		list_result.append(image_translator.translate_image_to_string(file_save_name))

		return list_result

	def check_black(self, list_data):
		# include all the alpha
		if len(list_data) == 4:

			# for i in range(len(list_data)-1):
			# 	if list_data[i] == 1:
			# 		return True
			if list_data[3] == 255:
				return True
			return False
		else:

			if 1 in list_data:
				return True
			else:
				return False