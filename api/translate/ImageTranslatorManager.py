import os
import uuid
from common.ImageTransparator import ImageTransparater
from common.ImageSegmentor import Segmentor

class ImageTranslatorManager:

    def __init__(self):
        return

    def translate_image(self, request, app):

        if request.method == "POST":
            file = request.files['file']
            encrypted_file_name = str(uuid.uuid4())
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], encrypted_file_name))
            image_transparator = ImageTransparater()
            file_transparent_name = image_transparator.start_transforming(encrypted_file_name)
            self.segmenting_image(file_transparent_name)
            return {"success": True}
        return {"success": False}

    def segmenting_image(self, file_name):

        segmentor = Segmentor()
        segmentor.start_segmenting_image(file_name)

