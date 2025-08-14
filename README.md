# Object Recognition using ResNet50

## 📌 Project Overview
This project demonstrates **object recognition** using the **ResNet50** deep learning architecture. ResNet50 (Residual Network with 50 layers) is a state-of-the-art convolutional neural network (CNN) pre-trained on the **ImageNet** dataset.  
The model is capable of classifying images into **1,000 object categories**, making it useful for a wide variety of computer vision tasks.

---

## 📂 Dataset
- **Source**: Pre-trained on the [ImageNet](https://image-net.org/) dataset.
- **Classes**: 1,000 object categories (e.g., animals, vehicles, tools, etc.)
- **Image Input Size**: 224 × 224 pixels (RGB).

---

## ⚙️ Project Workflow
1. **Import Libraries** – TensorFlow/Keras, NumPy, Matplotlib, PIL, etc.
2. **Load Pre-trained ResNet50 Model**
   - Weights: `"imagenet"`
   - Include top fully connected layers for classification
3. **Image Preprocessing**
   - Load an input image
   - Resize to 224×224 pixels
   - Convert to NumPy array
   - Expand dimensions to match model input
   - Preprocess using `preprocess_input` (ResNet50 specific)
4. **Prediction**
   - Pass the image through the model
   - Decode predictions using `decode_predictions` from Keras
   - Display top predicted classes with probabilities
5. **Visualization**
   - Show input image and prediction results

---

## 🧠 Model Architecture
- **Base Network**: ResNet50
- **Layers**: 50 convolutional + fully connected layers
- **Key Feature**: Residual connections to prevent vanishing gradients
- **Input Shape**: (224, 224, 3)
- **Output Shape**: (1,000 classes)

---
Performance depends on image clarity and similarity to ImageNet classes.

---

## 🚀 How to Run
1. Clone this repository or download the notebook.
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib pillow

jupyter notebook Object_Recognition_using_ResNet50.ipynb


