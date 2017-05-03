from common.ImageSegmentor import Segmentor
from common.GradientImageSegmentor import ImageSegmentor

if __name__ == "__main__":
	# first segmenting all the image first
	# file_name_path = "img/test2.jpg"
	# segmentor = Segmentor()
	# segmentor.start_segmenting_image(file_image_part=file_name_path)
	image_segmentor = ImageSegmentor("img/test1.jpg")
	image_segmentor.start_dsu_image_segmenting()