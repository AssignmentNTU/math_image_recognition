import os
import uuid
import thread
from common.ImageTransparator import ImageTransparater
from common.ImageSegmentor import Segmentor


class ImageTranslatorManager:

    def __init__(self):
        return

    def translate_image(self, request, app):

        if request.method == "POST":
            file = request.files['file']

            encrypted_file_name = str(uuid.uuid4()) + self.get_file_name_extension(file)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], encrypted_file_name))
            file.close()
            image_transparator = ImageTransparater()
            full_path_name = "/Users/edwardsujono/Python_Project/math_image_recognition/upload_image/" + encrypted_file_name
            # just for simple logs
            print("full_path_name: %s is being processed" % full_path_name)
            transparent_file_data= image_transparator.start_transforming(full_path_name, return_image=False)
            # transparent file date can be either file name or real file
            # return_image False means the return value would be the filename
            list_data = self.segmenting_image(transparent_file_data=transparent_file_data)
            return {"list_answer": list_data}
        return {"success": False}

    def segmenting_image(self, transparent_file_data):

        segmentor = Segmentor()
        list_file_name  = segmentor.start_segmenting_image(transparent_file_data, pass_image_name=True)
        return list_file_name

    def get_file_name_extension(self, file_data):
        file_name = file_data.filename
        str_file = file_name.split(".")
        file_extension = str_file[1]
        return "."+file_extension
