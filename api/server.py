import json

from flask import Flask, request, redirect, url_for

from api.translate.ImageTranslatorManager import ImageTranslatorManager



UPLOAD_FOLDER = "/Users/edwardsujono/Python_Project/math_image_recognition/upload_image"
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/api/translate_image", methods=['GET', 'POST'])
def get_translate_image():
    translator_manager = ImageTranslatorManager()
    result = translator_manager.translate_image(request=request, app=app)
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7070)
