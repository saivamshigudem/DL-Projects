# ============================================================
#  DATA UNDERSTANDING & IMAGE EXPLORATION
# PROJECT : Image Recognition using CNN
# FRAMEWORK : PyTorch
# DATASET : Intel Image Classification
# ============================================================


# ============================================================
# SECTION 1 : IMPORT LIBRARIES
# ============================================================

# CODE EXPLANATION:
# Import all required libraries for
# image processing and visualization.

import os
import random

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import pandas as pd


# ============================================================
# SECTION 2 : DATASET DIRECTORY
# ============================================================

# CODE EXPLANATION:
# Specify the dataset paths.

train_path = "seg_train"

test_path = "seg_test"

prediction_path = "seg_pred"

train_path = os.path.join("seg_train", "seg_train")
test_path = os.path.join("seg_test", "seg_test")
print("Prediction Folder :", prediction_path)


# ============================================================
# SECTION 3 : DISPLAY FOLDER STRUCTURE
# ============================================================

# CODE EXPLANATION:
# Display all class folders.

classes = sorted(os.listdir(train_path))

print("\nAvailable Classes")

for index, class_name in enumerate(classes):

    print(f"{index+1}. {class_name}")


# ============================================================
# SECTION 4 : TOTAL NUMBER OF CLASSES
# ============================================================

# CODE EXPLANATION:
# Count total classes.

print()

print("Total Classes :", len(classes))


# ============================================================
# SECTION 5 : NUMBER OF IMAGES PER CLASS
# ============================================================

# CODE EXPLANATION:
# Count images inside every folder.

image_count = {}

for class_name in classes:

    folder = os.path.join(train_path, class_name)

    total_images = len(os.listdir(folder))

    image_count[class_name] = total_images

print()

print(pd.DataFrame(

    image_count.items(),

    columns=["Class","Images"]

))


# ============================================================
# SECTION 6 : DISPLAY SAMPLE IMAGES
# ============================================================

# CODE EXPLANATION:
# Display one random image
# from every class.

plt.figure(figsize=(15,8))

for index, class_name in enumerate(classes):

    folder = os.path.join(train_path, class_name)

    image_name = random.choice(

        os.listdir(folder)

    )

    image_path = os.path.join(

        folder,

        image_name

    )

    image = Image.open(image_path)

    plt.subplot(2,3,index+1)

    plt.imshow(image)

    plt.title(class_name)

    plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# SECTION 7 : IMAGE DIMENSIONS
# ============================================================

# CODE EXPLANATION:
# Check image size for one sample
# from each class.

print()

print("Image Dimensions")

for class_name in classes:

    folder = os.path.join(train_path,class_name)

    image_name = random.choice(

        os.listdir(folder)

    )

    image_path = os.path.join(

        folder,

        image_name

    )

    image = Image.open(image_path)

    print(

        class_name,

        ":",

        image.size

    )


# ============================================================
# SECTION 8 : PIXEL VALUE ANALYSIS
# ============================================================

# CODE EXPLANATION:
# Display pixel range.

sample_class = classes[0]

sample_folder = os.path.join(

    train_path,

    sample_class

)

sample_image = random.choice(

    os.listdir(sample_folder)

)

sample_path = os.path.join(

    sample_folder,

    sample_image

)

image = Image.open(sample_path)

image_array = np.array(image)

print()

print("Minimum Pixel :", image_array.min())

print("Maximum Pixel :", image_array.max())

print("Image Shape :", image_array.shape)


# ============================================================
# SECTION 9 : CLASS DISTRIBUTION
# ============================================================

# CODE EXPLANATION:
# Plot number of images
# available in each class.

plt.figure(figsize=(8,5))

plt.bar(

    image_count.keys(),

    image_count.values()

)

plt.title("Class Distribution")

plt.xlabel("Class")

plt.ylabel("Number of Images")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# SECTION 10 : DATASET SUMMARY
# ============================================================

# CODE EXPLANATION:
# Display important dataset information.

print()

print("======================================")

print("DATASET SUMMARY")

print("======================================")

print("Dataset Name : Intel Image Classification")

print("Framework : PyTorch")

print("Total Classes :", len(classes))

print("Classes")

for class_name in classes:

    print("-", class_name)

print()

print("Training Images")

print(sum(image_count.values()))

print()

print("Dataset Ready For Preprocessing")

print()

# ============================================================
#  IMAGE PREPROCESSING & DATALOADER
# PROJECT : Image Recognition using CNN
# FRAMEWORK : PyTorch
# ============================================================


# ============================================================
# SECTION 1 : IMPORT PYTORCH LIBRARIES
# ============================================================

# CODE EXPLANATION:
# Import only PyTorch libraries required for
# preprocessing and loading image datasets.

import torch

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader


# ============================================================
# SECTION 2 : IMAGE TRANSFORMATIONS
# ============================================================

# CODE EXPLANATION:
# Apply preprocessing and augmentation.
#
# Resize Images
# Random Horizontal Flip
# Random Rotation
# Convert Image to Tensor
# Normalize Pixel Values

train_transform = transforms.Compose([

    transforms.Resize((150,150)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])


test_transform = transforms.Compose([

    transforms.Resize((150,150)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])


print("Image Transformations Created Successfully")


# ============================================================
# SECTION 3 : IMAGEFOLDER DATASET
# ============================================================

# CODE EXPLANATION:
# Load images using ImageFolder.

train_dataset = datasets.ImageFolder(

    root=train_path,

    transform=train_transform

)

test_dataset = datasets.ImageFolder(

    root=test_path,

    transform=test_transform

)


print()

print("Training Images :", len(train_dataset))

print("Testing Images :", len(test_dataset))


# ============================================================
# SECTION 4 : CLASS TO INDEX MAPPING
# ============================================================

# CODE EXPLANATION:
# Display how class names are converted
# into numerical labels.

print()

print("Class Mapping")

print(train_dataset.class_to_idx)


# ============================================================
# SECTION 5 : DATALOADER
# ============================================================

# CODE EXPLANATION:
# DataLoader loads mini-batches
# during CNN training.

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True,

    num_workers=0

)

test_loader = DataLoader(

    test_dataset,

    batch_size=32,

    shuffle=False,

    num_workers=0

)


print()

print("Training Batches :", len(train_loader))

print("Testing Batches :", len(test_loader))


# ============================================================
# SECTION 6 : INSPECT ONE BATCH
# ============================================================

# CODE EXPLANATION:
# Verify batch size and tensor shapes.

images, labels = next(iter(train_loader))

print()

print("Image Batch Shape")

print(images.shape)

print()

print("Label Batch Shape")

print(labels.shape)


# ============================================================
# SECTION 7 : DISPLAY IMAGE TENSOR
# ============================================================

# CODE EXPLANATION:
# Display one image after transformation.

image = images[0]

image = image.permute(1,2,0)

image = image.numpy()

image = (image * np.array([0.229,0.224,0.225])) + np.array([0.485,0.456,0.406])

image = np.clip(image,0,1)

plt.figure(figsize=(5,5))

plt.imshow(image)

plt.title(

    classes[labels[0]]

)

plt.axis("off")

plt.show()


# ============================================================
# SECTION 8 : DISPLAY MULTIPLE IMAGES
# ============================================================

# CODE EXPLANATION:
# Display first 9 images from a batch.

plt.figure(figsize=(10,10))

for i in range(9):

    image = images[i]

    image = image.permute(1,2,0)

    image = image.numpy()

    image = (image * np.array([0.229,0.224,0.225])) + np.array([0.485,0.456,0.406])

    image = np.clip(image,0,1)

    plt.subplot(3,3,i+1)

    plt.imshow(image)

    plt.title(classes[labels[i]])

    plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# SECTION 9 : SAVE CLASS NAMES
# ============================================================

# CODE EXPLANATION:
# Save class names for deployment.

import joblib

joblib.dump(

    classes,

    "class_names.pkl"

)

print()

print("Class Names Saved Successfully")


# ============================================================
# SECTION 10 : VERIFY DATASET
# ============================================================

# CODE EXPLANATION:
# Verify input size.

print()

print("Image Shape :", images.shape)

print("Number of Classes :", len(classes))

print("Classes")

for c in classes:

    print("-", c)


# ============================================================
# SECTION 11 : CPU CHECK
# ============================================================

# CODE EXPLANATION:
# Verify available device.

device = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)

print()

print("Running On :", device)


# ============================================================
# SECTION 12 : PART-2 SUMMARY
# ============================================================

print()

print("======================================")

print("✔ Image Transformations Applied")

print("✔ Data Augmentation Completed")

print("✔ ImageFolder Created")

print("✔ DataLoader Created")

print("✔ Batch Verification Completed")

print("✔ Class Mapping Generated")

print("✔ Class Names Saved")

print("✔ Ready For CNN Model Development")

# ============================================================
# CNN MODEL DEVELOPMENT & EVALUATION
# PROJECT : Image Recognition using CNN
# FRAMEWORK : PyTorch
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED MODULES
# ============================================================

# CODE EXPLANATION:
# Import only the modules required for
# model development and evaluation.

import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns


# ============================================================
# SECTION 2 : BUILD CNN MODEL
# ============================================================

# CODE EXPLANATION:
# CNN Architecture:
# Input Image (3x150x150)
# ↓
# Conv2D → ReLU → MaxPool
# ↓
# Conv2D → ReLU → MaxPool
# ↓
# Conv2D → ReLU → MaxPool
# ↓
# Flatten
# ↓
# Fully Connected Layer
# ↓
# Output Layer (6 Classes)

class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 18 * 18, 256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, len(classes))

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = CNNModel().to(device)

print(model)


# ============================================================
# SECTION 3 : LOSS FUNCTION
# ============================================================

# CODE EXPLANATION:
# CrossEntropyLoss is suitable
# for multi-class classification.

criterion = nn.CrossEntropyLoss()


# ============================================================
# SECTION 4 : OPTIMIZER
# ============================================================

# CODE EXPLANATION:
# Adam optimizer updates
# model parameters.

optimizer = optim.Adam(

    model.parameters(),

    lr=0.001

)


# ============================================================
# SECTION 5 : TRAINING SETTINGS
# ============================================================

epochs = 10

train_losses = []

train_accuracies = []


# ============================================================
# SECTION 6 : TRAIN CNN MODEL
# ============================================================

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)

    train_accuracies.append(epoch_accuracy)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# ============================================================
# SECTION 7 : LOSS CURVE
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(train_losses, marker="o")

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.show()


# ============================================================
# SECTION 8 : ACCURACY CURVE
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(train_accuracies, marker="o")

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.show()


# ============================================================
# SECTION 9 : MODEL EVALUATION
# ============================================================

model.eval()

all_predictions = []

all_labels = []

correct = 0

total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())


test_accuracy = 100 * correct / total

print()

print("Test Accuracy :", round(test_accuracy,2), "%")


# ============================================================
# SECTION 10 : CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions

)

plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=classes,

    yticklabels=classes

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()


# ============================================================
# SECTION 11 : CLASSIFICATION REPORT
# ============================================================

print()

print(classification_report(

    all_labels,

    all_predictions,

    target_names=classes

))


# ============================================================
# SECTION 12 : SAVE MODEL
# ============================================================

torch.save(

    model.state_dict(),

    "cnn_model.pth"

)

print()

print("CNN Model Saved Successfully")


# ============================================================
# SECTION 13 : SAMPLE PREDICTIONS
# ============================================================

images, labels = next(iter(test_loader))

images = images.to(device)

model.eval()

with torch.no_grad():

    outputs = model(images)

_, predictions = torch.max(outputs,1)

plt.figure(figsize=(12,8))

for i in range(6):

    image = images[i].cpu().permute(1,2,0).numpy()

    image = (image * np.array([0.229,0.224,0.225])) + np.array([0.485,0.456,0.406])

    image = np.clip(image,0,1)

    plt.subplot(2,3,i+1)

    plt.imshow(image)

    plt.title(

        f"Actual : {classes[labels[i]]}\nPredicted : {classes[predictions[i]]}"

    )

    plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# SECTION 14 : FINAL SUMMARY
# ============================================================

print()


print("=======================================")

print("CNN Model Built Successfully")

print("Model Trained Successfully")

print("Training Accuracy :", round(train_accuracies[-1],2), "%")

print("Testing Accuracy :", round(test_accuracy,2), "%")

print("Confusion Matrix Generated")

print("Classification Report Generated")

print("CNN Model Saved Successfully")

print()

print("Ready For Deployment")

# ============================================================
#  MODEL DEPLOYMENT USING FASTAPI
# PROJECT : Image Recognition using CNN
# FRAMEWORK : PyTorch
# ============================================================


# ============================================================
# SECTION 1 : SAVE APP.PY
# ============================================================

# CODE EXPLANATION:
# Create the FastAPI application file.
# This file loads the trained CNN model,
# preprocesses images,
# predicts the image class,
# and returns the prediction.

app_code = '''

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import joblib
import numpy as np


# -------------------------------
# Load Class Names
# -------------------------------

classes = joblib.load("class_names.pkl")


# -------------------------------
# CNN MODEL
# -------------------------------

class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel,self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128*18*18,256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256,len(classes))

        )

    def forward(self,x):

        return self.classifier(self.features(x))


model = CNNModel()

model.load_state_dict(

    torch.load(

        "cnn_model.pth",

        map_location=torch.device("cpu")

    )

)

model.eval()


transform = transforms.Compose([

    transforms.Resize((150,150)),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])


app = FastAPI(

    title="CNN Image Recognition API",

    version="1.0"

)


@app.get("/")

def home():

    return {

        "message":"CNN Image Recognition API"

    }


@app.post("/predict")

async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probability = torch.softmax(output,dim=1)

        confidence,prediction = torch.max(

            probability,

            dim=1

        )

    return {

        "Predicted_Class":classes[prediction.item()],

        "Confidence":round(float(confidence.item()),4)

    }

'''

print(app_code)


# ============================================================
# SECTION 2 : SAVE APP FILE
# ============================================================

with open("app.py","w") as file:

    file.write(app_code)

print("app.py Created Successfully")


# ============================================================
# SECTION 3 : CREATE REQUIREMENTS FILE
# ============================================================

requirements = '''

torch
torchvision
numpy
pandas
matplotlib
Pillow
joblib
fastapi
uvicorn
python-multipart

'''

with open("requirements.txt","w") as file:

    file.write(requirements)

print("requirements.txt Created Successfully")


# ============================================================
# SECTION 4 : PROJECT STRUCTURE
# ============================================================

print("""

Image_Recognition_CNN/

│

├── seg_train/

├── seg_test/

├── seg_pred/

├── cnn_model.pth

├── class_names.pkl

├── app.py

├── requirements.txt

└── Image_Recognition_Using_CNN.ipynb

""")


# ============================================================
# SECTION 5 : RUN FASTAPI
# ============================================================

print("""

Open Terminal

Run:

uvicorn app:app --reload

""")


# ============================================================
# SECTION 6 : OPEN SWAGGER UI
# ============================================================

print("""

Open Browser

http://127.0.0.1:8000/docs

""")


# ============================================================
# SECTION 7 : TEST API
# ============================================================

print("""

1. Click POST /predict

2. Click Try it out

3. Upload any JPG or PNG image

4. Click Execute

""")


# ============================================================
# SECTION 8 : SAMPLE RESPONSE
# ============================================================

print("""

{

    "Predicted_Class":"forest",

    "Confidence":0.9876

}

""")


# ============================================================
# SECTION 9 : INDUSTRY IMPROVEMENTS
# ============================================================

print("""

Future Improvements

✔ Transfer Learning (ResNet18)

✔ ResNet50

✔ EfficientNet

✔ MobileNetV3

✔ Vision Transformer (ViT)

✔ Grad-CAM Visualization

✔ TensorBoard Logging

✔ Docker Deployment

✔ AWS EC2 Deployment

✔ Streamlit Frontend

✔ ONNX Model Export

✔ Quantization for Faster CPU Inference

""")


# ============================================================
# SECTION 10 : FINAL PROJECT SUMMARY
# ============================================================

print()

print("===================================================")

print("END-TO-END CNN PROJECT COMPLETED SUCCESSFULLY")

print("===================================================")

print("✔ Data Understanding & EDA")

print("✔ Image Preprocessing")

print("✔ Data Augmentation")

print("✔ DataLoader")

print("✔ CNN Model Development")

print("✔ Model Training")

print("✔ Model Evaluation")

print("✔ Confusion Matrix")

print("✔ Classification Report")

print("✔ Model Saved (.pth)")

print("✔ FastAPI Deployment")

print("✔ Swagger API")

print("✔ Image Prediction API")

print()

print("Congratulations!")

print("You have successfully built an End-to-End")

print("Image Recognition System using CNN and PyTorch.")
