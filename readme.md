# Face Mask Detection using CNN (Transfer Learning)

## 📌 Project Overview
This project is a **binary image classification model** that detects whether a person is wearing a **mask or not** using a Convolutional Neural Network (CNN) with **transfer learning**.

---

## 📂 Dataset Structure

dataset/
│
├── train/
│   ├── mask/
│   └── no_mask/
│
├── val/
│   ├── mask/
│   └── no_mask/
│
├── test/
│   ├── mask/
│   └── no_mask/


- Train: ~5000 images per class  
- Validation: ~400 images per class  
- Test: ~500 images per class  

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 🧠 Model Details

- Base Model: Transfer Learning (EfficientNet / MobileNetV2)
- Input Size: 224x224
- Output: Binary (Mask / No Mask)
- Activation: Sigmoid
- Loss Function: Binary Crossentropy
- Optimizer: Adam

---

## 🔄 Training Steps

1. Data preprocessing and normalization
2. Data augmentation (rotation, zoom, flip)
3. Load pretrained model
4. Freeze base layers
5. Add custom classification head
6. Train model
7. Fine-tune top layers
8. Evaluate on test data

---

## 📊 Performance

- Accuracy: ~90% – 98% (depending on tuning)
- Good generalization on real-world images

---

## ▶️ How to Run

### 1. Install dependencies

```

pip install tensorflow opencv-python matplotlib

```

### 2. Train the model
Run the notebook:
```

CNN.ipynb

```

### 3. Predict on image
```

python predict.py --image test.jpg

```

### 4. Run real-time detection
```

python webcam_detection.py

```

---

## 💾 Model Saving

Model is saved as:
```

mask_detector_model.h5

```

Load using:
```

from tensorflow.keras.models import load_model
model = load_model("mask_detector_model.h5")

```

---

## 🎥 Real-Time Detection

- Uses OpenCV
- Detects face using Haar Cascade
- Classifies mask / no mask in real-time via webcam

---

## 🚀 Future Improvements

- Use better face detectors (MTCNN / DNN)
- Deploy using Streamlit or Flask
- Convert model to TensorFlow Lite (mobile)
- Improve dataset with edge cases (low light, blur)

---

## 📌 Author

Jash Sharma