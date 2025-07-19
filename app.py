import os
from flask import Flask, render_template, request
import numpy as np
import cv2
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = 'static/photos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and label encoder
model = joblib.load("model/tumor_model.pkl")
le = joblib.load("model/label_encoder.pkl")

IMG_SIZE = 100  # Same size used during training

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray.flatten().reshape(1, -1)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_img():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            print(f"✅ File saved at: {path}")

            try:
                features = preprocess_image(path)
                prediction = model.predict(features)
                confidence = model.predict_proba(features).max()
                label = le.inverse_transform(prediction)[0]

                print(f"✅ Prediction: {label}, Confidence: {confidence}")

                return render_template('result.html',
                                       label=label,
                                       confidence=round(confidence * 100, 2),
                                       file_path=filename)  # ✅ Only send filename
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
                return "Error during prediction. Check console."
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
