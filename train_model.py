import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump
from collections import Counter

# --- 1️⃣ Define dataset path ---
DATA_DIR = "data"  # Folder with subfolders 'ripe', 'unripe', 'overripe'

# --- 2️⃣ Prepare image data ---
print("🔍 Scanning dataset folders...")
X, y = [], []
labels = os.listdir(DATA_DIR)
print("📁 Found label folders:", labels)

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Could not read {img_path}")
                continue
            img = cv2.resize(img, (100, 100))
            avg_color = img.mean(axis=(0, 1))  # Average RGB color
            X.append(avg_color)
            y.append(label)

print(f"📸 Total images found: {len(X)}")

if len(X) == 0:
    raise ValueError("❌ No images found! Please add some images inside data folders.")

# --- 3️⃣ Check class distribution ---
counts = Counter(y)
print("📊 Image count per class:", dict(counts))

min_class_count = min(counts.values())
num_classes = len(counts)

# --- 4️⃣ Auto adjust test_size based on dataset size ---
if len(X) < num_classes * 2:
    print("⚠️ Not enough images per class for stratified split.")
    print("➡️ Temporarily disabling stratify and using 80-20 split.")
    stratify_option = None
else:
    stratify_option = y

test_size = max(0.2, num_classes / len(X))  # ensure test set has enough samples

# --- 5️⃣ Train-test split ---
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_option
    )
except ValueError as e:
    print("⚠️ Stratified split failed, using non-stratified split instead.")
    print(f"   Error details: {e}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

# --- 6️⃣ Train model ---
print("🧠 Training model...")
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# --- 7️⃣ Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Training complete! Accuracy: {acc*100:.2f}%")

# --- 8️⃣ Save model ---
os.makedirs("models", exist_ok=True)
dump(model, "models/ripeness_model.joblib")
print("💾 Model saved as models/ripeness_model.joblib")
