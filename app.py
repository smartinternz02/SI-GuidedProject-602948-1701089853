from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model("asl_model.h5")
def preprocess_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, (64, 64))
    img_array = img_array.reshape(1, 64, 64, 1)
    img_array = img_array / 255.0
    return img_array
@app.route('/')
def index():
    return render_template('index.html', prediction=None)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')
        image_path = "uploads/input_image.jpg"
        file.save(image_path)
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        predicted_letter = chr(predicted_class + ord('A')) if predicted_class < 26 else \
                           'del' if predicted_class == 26 else \
                           'nothing' if predicted_class == 27 else \
                           'space'
        return render_template('index.html', prediction=f'Predicted Letter: {predicted_letter}')
    except Exception as e:
        return render_template('index.html', prediction='Error processing image')
if __name__ == '__main__':
    app.run(debug=True)
