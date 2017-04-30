import numpy as np
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage import io
from PIL import Image

# This class will brute force the image contour and crop it


class Segmentor:

	# LIMIT NUMBER GAP MEANS THAT TOTAL COLUMN 0
	# IN THE IMAGE
	LIMIT_NUMBER_GAP = 1

	def __init__(self):
		return

	def start_segmenting_image(self, file_image_part):

		image = io.imread(file_image_part)
		image_pil = Image.open(file_image_part)
		result_boundaries = find_boundaries(image, mode='outer').astype(np.uint8)

		x_range = image.shape[1]		# width
		y_range = image.shape[0]		# height

		list_x = []
		list_y = []

		number_gap = 0
		number_part = 0

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
				elif len(image.shape) == 3:
					if 1 in result_boundaries[j][i]:
						list_y.append(j)
						list_x.append(i)
						gap = True

			if not gap:
				number_gap += 1

			if number_gap == self.LIMIT_NUMBER_GAP:

				number_gap = 0

				if len(list_x) == 0 or len(list_y) == 0:
					continue

				min_x = min(list_x)
				max_x = max(list_x)

				min_y = min(list_y)
				max_y = max(list_y)

				image_part = image_pil.crop((min_x, min_y, max_x, max_y ))
				image_part.save("image_"+str(number_part)+".png")

				number_part += 1

				list_x = []
				list_y = []

			# last one usually there is no gap in between

		if len(list_x) == 0 or len(list_y) == 0:
			return

		min_x = min(list_x)
		max_x = max(list_x)

		min_y = min(list_y)
		max_y = max(list_y)

		image_part = image_pil.crop((min_x, min_y, max_x, max_y ))
		image_part.save("image_"+str(number_part)+".png")
