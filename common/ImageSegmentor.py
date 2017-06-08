import numpy as np
import uuid
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage import io, img_as_ubyte
from skimage.transform import resize
from PIL import Image
from ImageTranslator import ImageTranslator
from ImageTransparator import ImageTransparater
# This class will brute force the image contour and crop it


class Segmentor:

	# LIMIT NUMBER GAP MEANS THAT TOTAL COLUMN 0
	# IN THE IMAGE
	LIMIT_NUMBER_GAP = 10
	DIRECTORY_PATH = "/Users/edwardsujono/Python_Project/math_image_recognition/img/img_segmenting/"
	LIMIT_BLACK_PIXEL = 50

	def __init__(self):
		self.image_translator = ImageTranslator()
		self.image_transparater = ImageTransparater()
		return

	def start_segmenting_image(self, file_image_part, pass_image_name=True, file_non_transparent=None):

		list_result = []

		if pass_image_name:
			image = io.imread(file_image_part)
			image_to_cut = io.imread(file_non_transparent)
		else:
			image = file_image_part

		image_pil = Image.open(file_image_part) if file_non_transparent is None else Image.open(file_non_transparent)
		result_boundaries = find_boundaries(image, mode='outer').astype(np.uint8)

		x_range = image.shape[1]		# width
		y_range = image.shape[0]		# height

		list_x = []
		list_y = []

		number_gap = 0
		number_part = 0

		get_pixel_start = False
		count_black_pixel = 0
		for i in range(x_range):
			# gap means that it is different segment of image
			gap = False

			for j in range(y_range):
				# this is gray scale image
				if len(image.shape) == 2:
					if result_boundaries[j][i] == 1:
						list_y.append(j)
						list_x.append(i)
						get_pixel_start = True
						count_black_pixel += 1
						gap = True
				# this is RGB value
				else:
					if self.check_black(image[j][i]):
						list_y.append(j)
						list_x.append(i)
						count_black_pixel += 1
						get_pixel_start = True
						gap = True

			if not gap and get_pixel_start:
				number_gap += 1

			# to reduce possible noise
			if number_gap >= self.LIMIT_NUMBER_GAP and count_black_pixel <= self.LIMIT_BLACK_PIXEL:
				list_x = []
				list_y = []

			# start to crop the image
			if number_gap >= self.LIMIT_NUMBER_GAP and count_black_pixel > self.LIMIT_BLACK_PIXEL:
				count_black_pixel = 0
				number_gap = 0

				if len(list_x) == 0 or len(list_y) == 0:
					continue

				min_x = min(list_x)
				max_x = max(list_x)

				min_y = min(list_y)
				max_y = max(list_y)

				if min_x == max_x or min_y == max_y:
					list_x = []
					list_y = []
					continue

				image_part = image_pil.crop((min_x, min_y, max_x, max_y)).resize((45, 45), Image.ANTIALIAS)
				# # image_part = resize(image[min_x:max_x, min_y:max_y], (45, 45), mode="reflect")
				# file_save_name = self.DIRECTORY_PATH + str(uuid.uuid4()) + "_number_part_" + str(number_part) + ".jpg"
				# print ("file_save_name:%s" % file_save_name)
				# image_part = img_as_ubyte(image_part)
				# image_part.save(file_save_name)
				# image_part.close()
				# # io.imsave(file_save_name, image_part)
				# list_result.append(self.image_translator.translate_image_to_string(file_image_part))
				# image_part = image_pil.thumbnail((min_x, min_y, max_x, max_y), Image.ANTIALIAS)
				# image_part = resize(image_to_cut[min_x:max_x, min_y:max_y], (45, 45), mode="reflect")
				# image_part = image[min_x:max_x, min_y:max_y], (45, 45), mode="reflect")
				file_save_name = self.DIRECTORY_PATH + str(uuid.uuid4()) + "_number_part_" + str(number_part) + ".jpg"
				# image_part = img_as_ubyte(image_part)
				print ("file_save_name:%s" % file_save_name)

				image_part = image_part.convert("L")
				image_part.save(file_save_name)
				image_part.close()
				# io.imsave(file_save_name, image_part)
				image_transform_white = self.image_transparater.start_transforming_back_to_jpg(file_save_name, return_array=False)
				print ("file_save_white:%s" % image_transform_white)
				list_result.append(self.image_translator.translate_image_to_string(image_transform_white))

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

		image_part = image_pil.crop((min_x, min_y, max_x, max_y)).resize((45, 45), Image.ANTIALIAS)
		# image_part = image_pil.thumbnail((min_x, min_y, max_x, max_y), Image.ANTIALIAS)
		# image_part = resize(image_to_cut[min_x:max_x, min_y:max_y], (45, 45), mode="reflect")
		# image_part = image[min_x:max_x, min_y:max_y], (45, 45), mode="reflect")
		file_save_name = self.DIRECTORY_PATH + str(uuid.uuid4()) + "_number_part_" + str(number_part) + ".jpg"
		image_part = image_part.convert("L")
		# image_part = img_as_ubyte(image_part)
		print ("file_save_name:%s" % file_save_name)
		image_part.save(file_save_name)
		image_part.close()
		# io.imsave(file_save_name, image_part)
		image_transform_white = self.image_transparater.start_transforming_back_to_jpg(file_save_name, return_array=False)
		print ("file_save_white:%s" % image_transform_white)
		list_result.append(self.image_translator.translate_image_to_string(image_transform_white))

		return list_result

	def check_black(self, list_data):
		# include all the alpha
		if len(list_data) == 4:

			# for i in range(len(list_data)-1):
			# 	if list_data[i] == 1:
			# 		return True
			# black
			if list_data[3] == 255:
				return True
			return False
		else:

			if 1 in list_data:
				return True
			else:
				return False