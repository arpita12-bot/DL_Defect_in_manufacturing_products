#  🎯 Detect-in-manufacturing-products-from-images
 ---
CASTING PRODUCT QUALITY INSPECTION:<br>

Casting is a manufacturing process in which liquid material is poured into a mold to solidify. Many types of defects or unwanted irregularities can occur during this process. The industry has its quality inspection department to remove defective products from the production line, but this is very time consuming since it is carried out manually. Furthermore, there is a chance of misclassifying due to human error, causing rejection of the whole product order.<br>

💻 Use CNN(CONVOLUTIONAL NEURAL NETWORK) deep learning model to predict defects on steel sheets and visually localize the defect using VGG-like model.

###  ✅ Overview
---
This project aims to predict surface defects on steel sheets from images.
* The goal is to build a deep learning model to detect casting faults using an image dataset from Kaggle.
* The dataset contains ~10,500 images divided into two classes — faulty and okay parts.
* The primary aim is image classification, with a focus on:
* Achieving high accuracy with a simple CNN model.
* Balancing model complexity (parameters) and training time.
1. What’s the simplest CNN model that can achieve high accuracy?<br>
   + Initially considered models like ResNet models.<br>
   + However, these models have millions of parameters (e.g., >20M), which is excessive for this dataset size.<br>
   + A simpler CNN architecture can achieve comparable performance with much fewer parameters. So we use CNN model like VGG model.
3. How to balance accuracy and the number of parameters?<br>
   + The aim is to find a model that offers optimal performance while being computationally efficient.<br>
   + Once a good model is found, explore reducing the number of parameters without a significant accuracy drop.<br>
   + This helps understand the trade-off between model complexity and performance.<br>

The VGG-like CNN model is the best choice for this fault detection task because:<br>
1. It achieves high accuracy with reasonable training time.<br>
2. It follows the core design principles of VGG (using small 3×3 filters).<br>
3. It is simpler, faster, and easier to train compared to complex architectures.
<img width="226" height="102" alt="image" src="https://github.com/user-attachments/assets/4ea42f29-ba42-4ce6-939a-48b9ddbe2e82" />


### ✅ Project description
---
let us automate the inspection process by training top-view images of a casted steel manufacturing products using custom method with Convolutional Neural Network (CNN) so that it can distinguish accurately between defect from the ok one.<br>

We will break down into several steps:<br>
1. Load the images and apply the data augmentation technique
2. Visualize the images
3. Training with validation: define the architecture, compile the model, model fitting and evaluation
4. Testing on unseen images
5. Make a conclusion


### 📁 Project Structure
---
<img width="526" height="320" alt="image" src="https://github.com/user-attachments/assets/f8f705ea-33d7-42ca-aafd-068d6311bd53" />

### 🧠 Model Information
---
- **Architecture:** Transfer Learning using VGG-like model 
- **Frameworks:** PyTorch
- **Dataset:**  
  - 3146 images without defective
  - 4211 images defective
- **Input size:** 224 × 224 × 3  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam (`learning_rate=1e-4`)
- **Epochs:** 20  
- **Batch Size:** 16
<img width="341" height="247" alt="image" src="https://github.com/user-attachments/assets/3c375151-2396-41f7-9555-8506273b8482" />


### 🚀 Training the Model
Run the following command to train the model:
```bash
python train_model.py 
```
This will:
1. Load and preprocess the dataset  
2. Train Simple CNN VGG-like model head layers
  <img width="497" height="281" alt="image" src="https://github.com/user-attachments/assets/1961555e-e9ad-41da-b2e0-3aa64bf66938" />

3. Save the model as `best_model.pth`  
4. Generate a training plot `plot.png`
<img width="376" height="281" alt="image" src="https://github.com/user-attachments/assets/2be3754d-4772-4d88-a154-9165cd4aa4db" />
   

---


### ⭐ Defect Detection
Once the model is trained, run:
```bash
python evaluate.py
```
This will:
- Start the image processing.
- Detect defects using VGG like model on steel manufactured casted products.
- Classify each detected product as **Ok** ✅ or **defected** ❌  and show accuracy report.
<img width="900" height="330" alt="image" src="https://github.com/user-attachments/assets/7dc4d177-4bc1-4298-95b3-f75976f40300" />

<img width="566" height="353" alt="image" src="https://github.com/user-attachments/assets/d0581dfd-af21-4f08-9d90-9664cdabd8d5" />





### 🧩 Dependencies
Install required packages:
```bash
!pip install torchsummary -q
!pip install torcheval -q
!pip install grad-cam -q
!pip install imgaug
!pip matplotlib
```
### 🧠 How It Works
1. **Defect Detection:** Uses a pre-trained CNN-based model (VGG like model) to detect regions on steel surfaces that may contain defects such as cracks, scratches, inclusions, or dents.The model scans the input image frame-by-frame or from stored samples.
2. **Defect Classification:** The model classifies each region into categories like “Defective” or “Non-Defective”, or further into defect types (e.g., Crack, Scratch, Burr, Hole).
3. **Real-time Prediction:** The system performs real-time analysis on images from the production line. we visualize the results by comparing its true label with the predicted label and also provide the probability of each image being on the predicted class. A blue color on the text indicates that our model correctly classify the image, otherwise red color is used.

---
### ⚖️ Sample Results
| With Defect |<br>
|------------|------------|
| ✅ 98.3% Accuracy 

---
### 🚧 Future Improvements
- Extend the model to detect defects from videos and real-time factory streams by enabling real-time factory monitoring through CCTV integration.
- Deploying the model to **Edge devices** or **Raspberry Pi**.
- Track model drift and performance metrics during live deployment.
- Integrate PCA-based dimensionality reduction for faster and more efficient image processing.
  
### 👨‍💻 Author
**Mohit Sharma(M25DE1001) and  Arpita Kundu(M25DE1004)**  
_MTech Data Engineering, IIT Jodhpur_  




































