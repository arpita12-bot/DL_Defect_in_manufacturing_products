#  🎯 Detect-in-manufacturing-products-from-images

Use CNN(CONVOLUTIONAL NEURAL NETWORK) deep learning model to predict defects on steel sheets and visually localize the defect using VGG-like model.

##  ✅ Overview
This project aims to predict surface defects on steel sheets from images.
* The goal is to build a deep learning model to detect casting faults using an image dataset from Kaggle.
* The dataset contains ~10,500 images divided into two classes — faulty and okay parts.
* The primary aim is image classification, with a focus on:
* Achieving high accuracy with a simple CNN model.
* Balancing model complexity (parameters) and training time.
1. What’s the simplest CNN model that can achieve high accuracy?<br>
   Initially considered models like ResNet models.<br>
   However, these models have millions of parameters (e.g., >20M), which is excessive for this dataset size.<br>
   * A simpler CNN architecture can achieve comparable performance with much fewer parameters. So we use CNN model like VGG model.
3. How to balance accuracy and the number of parameters?<br>
   The aim is to find a model that offers optimal performance while being computationally efficient.<br>
   Once a good model is found, explore reducing the number of parameters without a significant accuracy drop.<br>
   This helps understand the trade-off between model complexity and performance.<br>

The VGG-like CNN model is the best choice for this fault detection task because:<br>
1. It achieves high accuracy with reasonable training time.<br>
2. It follows the core design principles of VGG (using small 3×3 filters).<br>
3. It is simpler, faster, and easier to train compared to complex architectures.
---

## ✅ Project description

---

## 📁 Project Structure
Face-Mask-Detection/
│
├── train_mask_detector.py        # Train MobileNetV2 model
├── detect_mask_video.py          # Real-time mask detection via webcam
├── dataset/                      # Contains 'with_mask' and 'without_mask' subfolders
│   ├── with_mask/
│   └── without_mask/
├── mask_detector.keras           # Trained model file (output)
└── plot.png                      # Training accuracy/loss plot


## 🧠 Model Information
---
- **Architecture:** Transfer Learning using MobileNetV2 (pre-trained on ImageNet)
- **Frameworks:** TensorFlow, Keras, OpenCV
- **Dataset:**  
  - 2165 images with mask  
  - 1930 images without mask
- **Input size:** 224 × 224 × 3  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam (`learning_rate=1e-4`)
- **Epochs:** 20  
- **Batch Size:** 32


## 🚀 Training the Model
Run the following command to train the model:
```bash
python train_mask_detector.py --dataset dataset
```
This will:
1. Load and preprocess the dataset  
2. Train MobileNetV2 head layers  
3. Save the model as `mask_detector.keras`  
4. Generate a training plot `plot.png`  

---


## ⭐ Real-Time Detection
Once the model is trained, run:
```bash
python detect_mask_video.py
```
This will:
- Start the webcam feed  
- Detect faces using OpenCV’s DNN-based face detector  
- Classify each detected face as **Mask** 😷 or **No Mask** ❌  

Press **'q'** to quit the video stream.
## 🧩 Dependencies
Install required packages:
```bash
pip install tensorflow==2.11.0
pip install imutils scikit-learn matplotlib opencv-python
```
## 🧠 How It Works
1. **Face Detection:** Uses OpenCV’s pretrained Caffe model (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`) to locate faces.
2. **Mask Classification:** Each detected face is resized to 224×224 and passed into the MobileNetV2-based CNN for classification.
3. **Real-time Prediction:** Detected faces are highlighted in green (Mask) or red (No Mask) with confidence scores.

---
## ⚖️ Sample Results
| With Mask | Without Mask |
|------------|---------------|
| ✅ 98.3% Accuracy | ❌ 97.1% Accuracy |

---
## 🚧 Future Improvements
- Fine-tuning the MobileNetV2 base layers for higher accuracy.
- Deploying the model to **Edge devices** or **Raspberry Pi**.
- Integrating it with CCTV or live video monitoring systems.
- 
## 👨‍💻 Author
**Mohit Sharma(M25DE1001), Arpita Kundu(M25DE1004)**  
_MTech Data Engineering, IIT Jodhpur_  




