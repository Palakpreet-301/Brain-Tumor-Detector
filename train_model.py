import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Define dataset path
data_dir = "Dataset"
categories = os.listdir(data_dir)
IMG_SIZE = 100

data = []
labels = []

# Load and preprocess images
for category in categories:
    folder_path = os.path.join(data_dir, category)
    for filename in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(gray.flatten())
            labels.append(category)
        except Exception as e:
            print(f"Skipped {img_path} due to error: {e}")

# Encode labels
X = np.array(data)
le = LabelEncoder()
y = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Create model folder if not exist
os.makedirs("model", exist_ok=True)

# Save model and label encoder
joblib.dump(model, "model/tumor_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("✅ Model training complete and saved.")
print("📊 Evaluation Report:")
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))
