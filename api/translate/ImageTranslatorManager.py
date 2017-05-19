import os


class ImageTranslatorManager:

    def __init__(self):
        return

    def translate_image(self, request, app):

        if request.method == "POST":
            file = request.files['file']
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return {"success": True}
        return {"success": False}
