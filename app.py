from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif'}

model_path = 'D:/DEHAZE/finalmodel.h5'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def dehaze_image(image_path, model):
    img = Image.open(image_path)
    img = img.resize((256, 256))  
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    result_image = Image.fromarray((result[0] * 255).astype(np.uint8))
    return result_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        model = tf.keras.models.load_model(model_path)
        result_image = dehaze_image(file_path, model)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
        result_image.save(result_path)
        return redirect(url_for('show_result'))
    else:
        return redirect(request.url)

@app.route('/result')
def show_result():
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
    return render_template('result.html', result=result_path)

if __name__ == '__main__':
    app.run(debug=True)
