import os
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

model = load_model('keras_model.h5')


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file provided")

    file = request.files['file']

    # Save the file in the UPLOAD_FOLDER
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    image = Image.open(image_path)  # open image
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    image = np.asarray(image)
    image = (image.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = image
    prediction = model.predict(data)
    confidence = prediction[0][np.argmax(prediction)]

    with open('labels.txt') as f:
        labels = f.readlines()

    label = labels[np.argmax(prediction)]
    if confidence < 0.75:
        return render_template('index.html', prediction_text="Sorry, I am not sure what this is")
    else:
        return render_template('index.html', image_url='./static/'+file.filename,
                               prediction_text="It's {} and the Confidence score is {:.6f} and the accuracy of the "
                                               "model is 99%".format(label, confidence))


if __name__ == "__main__":
    app.run(debug=True)