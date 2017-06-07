import os
import uuid
import pickle
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
            full_path_name = os.path.join(app.config['UPLOAD_FOLDER'], encrypted_file_name)
            # just for simple logs
            print("full_path_name: %s is being processed" % full_path_name)
            transparent_file_data= image_transparator.start_transforming(full_path_name, return_image=False)
            # transparent file date can be either file name or real file
            # return_image False means the return value would be the filename
            list_data = self.segmenting_image(transparent_file_data=transparent_file_data, real_data=full_path_name)
            return {"list_answer": list_data}
        return {"success": False}

    def segmenting_image(self, transparent_file_data, real_data=None):

        segmentor = Segmentor()
        list_data = segmentor.start_segmenting_image(transparent_file_data, pass_image_name=True, file_non_transparent=real_data)
        return self.translate_to_label_from_number(list_data)

    def translate_to_label_from_number(self, list_data):
        with open("key_args.pickle", "rb") as handle:
            save_args = pickle.load(handle)

        list_label = []

        for data in list_data:
            list_label.append(save_args[data])

        return list_label

    def get_file_name_extension(self, file_data):
        file_name = file_data.filename
        str_file = file_name.split(".")
        file_extension = str_file[1]
        return "."+file_extension
