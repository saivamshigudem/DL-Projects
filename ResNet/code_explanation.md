
# ============================================================
# Medical Imaging using ResNet50
#  Data Loading & Exploratory Data Analysis
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import os

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Dataset Path
# ============================================================

dataset_path = "chest_xray"

train_path = os.path.join(

    dataset_path,

    "train"

)

test_path = os.path.join(

    dataset_path,

    "test"

)

val_path = os.path.join(

    dataset_path,

    "val"

)

print("\n")

print("="*70)

print("Dataset Path Loaded Successfully")

print("="*70)

# ============================================================
# Classes
# ============================================================

classes = os.listdir(

    train_path

)

print("\n")

print("="*70)

print("Classes")

print("="*70)

print(classes)

# ============================================================
# Count Images
# ============================================================

train_counts = {}

for category in classes:

    folder = os.path.join(

        train_path,

        category

    )

    train_counts[category] = len(

        os.listdir(folder)

    )

print("\n")

print("="*70)

print("Training Images")

print("="*70)

display(

    pd.DataFrame(

        train_counts.items(),

        columns=[

            "Class",

            "Images"

        ]

    )

)

# ============================================================
# Total Images
# ============================================================

train_total = sum(

    train_counts.values()

)

print("\n")

print("="*70)

print("Total Training Images")

print("="*70)

print(train_total)

# ============================================================
# Class Distribution
# ============================================================

plt.figure(

    figsize=(6,5)

)

sns.barplot(

    x=list(train_counts.keys()),

    y=list(train_counts.values())

)

plt.title(

    "Training Class Distribution"

)

plt.xlabel(

    "Class"

)

plt.ylabel(

    "Images"

)

plt.show()

# ============================================================
# Display Sample Images
# ============================================================

plt.figure(

    figsize=(10,5)

)

for i, category in enumerate(classes):

    image_name = os.listdir(

        os.path.join(

            train_path,

            category

        )

    )[0]

    image_path = os.path.join(

        train_path,

        category,

        image_name

    )

    image = cv2.imread(

        image_path

    )

    image = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )

    plt.subplot(

        1,

        2,

        i+1

    )

    plt.imshow(

        image

    )

    plt.title(

        category

    )

    plt.axis("off")

plt.show()

# ============================================================
# Image Resolution Analysis
# ============================================================

heights = []

widths = []

for category in classes:

    folder = os.path.join(

        train_path,

        category

    )

    for img_name in os.listdir(folder)[:100]:

        img = cv2.imread(

            os.path.join(

                folder,

                img_name

            )

        )

        if img is not None:

            h, w, _ = img.shape

            heights.append(h)

            widths.append(w)

print("\n")

print("="*70)

print("Image Resolution Statistics")

print("="*70)

print(

    "Average Height :", round(

        np.mean(heights),

        2

    )

)

print(

    "Average Width :", round(

        np.mean(widths),

        2

    )

)

print(

    "Minimum Height :", min(

        heights

    )

)

print(

    "Maximum Height :", max(

        heights

    )

)

print(

    "Minimum Width :", min(

        widths

    )

)

print(

    "Maximum Width :", max(

        widths

    )

)

# ============================================================
# Pixel Distribution
# ============================================================

sample_image = cv2.imread(

    image_path,

    cv2.IMREAD_GRAYSCALE

)

plt.figure(

    figsize=(8,5)

)

plt.hist(

    sample_image.ravel(),

    bins=50

)

plt.title(

    "Pixel Intensity Distribution"

)

plt.xlabel(

    "Pixel Value"

)

plt.ylabel(

    "Frequency"

)

plt.show()

# ============================================================
# Image Shape
# ============================================================

print("\n")

print("="*70)

print("Sample Image Shape")

print("="*70)

print(sample_image.shape)

# ============================================================
# Dataset Summary
# ============================================================

summary = pd.DataFrame({

    "Parameter":[

        "Dataset",

        "Classes",

        "Training Images",

        "Testing Folder",

        "Validation Folder"

    ],

    "Value":[

        "Chest X-Ray",

        len(classes),

        train_total,

        test_path,

        val_path

    ]

})

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

display(summary)

# ============================================================
#  Image Preprocessing & Data Augmentation
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("\n")

print("="*70)

print("TensorFlow Version")

print("="*70)

print(tf.__version__)

# ============================================================
# Image Configuration (CPU Optimized)
# ============================================================

IMAGE_HEIGHT = 128

IMAGE_WIDTH = 128

BATCH_SIZE = 16

print("\n")

print("="*70)

print("Image Configuration")

print("="*70)

print("Image Size :", IMAGE_HEIGHT, "x", IMAGE_WIDTH)

print("Batch Size :", BATCH_SIZE)

# ============================================================
# Training Image Generator
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=10,

    zoom_range=0.10,

    width_shift_range=0.10,

    height_shift_range=0.10,

    horizontal_flip=True

)

# ============================================================
# Validation Generator
# ============================================================

validation_datagen = ImageDataGenerator(

    rescale=1./255

)

# ============================================================
# Test Generator
# ============================================================

test_datagen = ImageDataGenerator(

    rescale=1./255

)

# ============================================================
# Train Generator
# ============================================================

train_generator = train_datagen.flow_from_directory(

    train_path,

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True

)

# ============================================================
# Validation Generator
# ============================================================

validation_generator = validation_datagen.flow_from_directory(

    val_path,

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False

)

# ============================================================
# Test Generator
# ============================================================

test_generator = test_datagen.flow_from_directory(

    test_path,

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False

)

print("\n")

print("="*70)

print("Image Generators Created Successfully")

print("="*70)

# ============================================================
# Class Indices
# ============================================================

print("\n")

print("="*70)

print("Class Indices")

print("="*70)

print(

    train_generator.class_indices

)

# ============================================================
# Dataset Statistics
# ============================================================

print("\n")

print("="*70)

print("Dataset Statistics")

print("="*70)

print(

    "Training Images :",

    train_generator.samples

)

print(

    "Validation Images :",

    validation_generator.samples

)

print(

    "Testing Images :",

    test_generator.samples

)

# ============================================================
# Display One Batch Shape
# ============================================================

images, labels = next(

    train_generator

)

print("\n")

print("="*70)

print("Batch Information")

print("="*70)

print(

    "Image Batch Shape :",

    images.shape

)

print(

    "Label Shape :",

    labels.shape

)

# ============================================================
# Display Sample Images
# ============================================================

plt.figure(

    figsize=(12,8)

)

for i in range(8):

    plt.subplot(

        2,

        4,

        i+1

    )

    plt.imshow(

        images[i]

    )

    plt.title(

        "NORMAL"

        if labels[i] == 0

        else

        "PNEUMONIA"

    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# ============================================================
# Pixel Range Verification
# ============================================================

print("\n")

print("="*70)

print("Pixel Value Verification")

print("="*70)

print(

    "Minimum Pixel :",

    np.min(images)

)

print(

    "Maximum Pixel :",

    np.max(images)

)

# ============================================================
# Dataset Information
# ============================================================

dataset_information = pd.DataFrame({

    "Parameter":[

        "Training Images",

        "Validation Images",

        "Testing Images",

        "Image Size",

        "Batch Size",

        "Classes"

    ],

    "Value":[

        train_generator.samples,

        validation_generator.samples,

        test_generator.samples,

        "128 x 128",

        BATCH_SIZE,

        train_generator.num_classes

    ]

})

print("\n")

print("="*70)

print("Dataset Information")

print("="*70)

display(

    dataset_information

)

# ============================================================
# Calculate Steps Per Epoch
# ============================================================

train_steps = train_generator.samples // BATCH_SIZE

validation_steps = validation_generator.samples // BATCH_SIZE

test_steps = test_generator.samples // BATCH_SIZE

print("\n")

print("="*70)

print("Training Configuration")

print("="*70)

print(

    "Train Steps :", train_steps

)

print(

    "Validation Steps :",

    validation_steps

)

print(

    "Test Steps :",

    test_steps

)

# ============================================================
# Dataset Ready
# ============================================================

print("\n")

print("="*70)

print("Dataset Ready for ResNet50")

print("="*70)

# ============================================================
#  ResNet50 Model Building & Evaluation
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import tensorflow as tf

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

# ============================================================
# Load Pretrained ResNet50
# ============================================================

base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_shape=(128,128,3)

)

print("\n")

print("="*70)

print("Pretrained ResNet50 Loaded Successfully")

print("="*70)

# ============================================================
# Freeze Base Model
# ============================================================

for layer in base_model.layers:

    layer.trainable = False

print("\n")

print("="*70)

print("Base Model Frozen")

print("="*70)

# ============================================================
# Build Custom Classification Head
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(

    128,

    activation="relu"

)(x)

x = Dropout(

    0.30

)(x)

output = Dense(

    1,

    activation="sigmoid"

)(x)

model = Model(

    inputs=base_model.input,

    outputs=output

)

print("\n")

print("="*70)

print("Custom Classification Head Added")

print("="*70)

# ============================================================
# Model Summary
# ============================================================

model.summary()

# ============================================================
# Compile Model
# ============================================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

print("\n")

print("="*70)

print("Model Compiled Successfully")

print("="*70)

# ============================================================
# Callbacks
# ============================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=2,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "best_resnet50_model.keras",

    monitor="val_accuracy",

    save_best_only=True

)

# ============================================================
# Train Model
# ============================================================

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=5,

    callbacks=[

        early_stop,

        checkpoint

    ],

    verbose=1

)

print("\n")

print("="*70)

print("Model Training Completed")

print("="*70)

# ============================================================
# Evaluate Model
# ============================================================

loss, accuracy = model.evaluate(

    test_generator,

    verbose=0

)

print("\n")

print("="*70)

print("Test Accuracy")

print("="*70)

print(

    round(

        accuracy,

        4

    )

)

# ============================================================
# Predictions
# ============================================================

prediction_probability = model.predict(

    test_generator,

    verbose=0

)

predictions = (

    prediction_probability > 0.5

).astype(int)

true_labels = test_generator.classes

# ============================================================
# Evaluation Metrics
# ============================================================

precision = precision_score(

    true_labels,

    predictions

)

recall = recall_score(

    true_labels,

    predictions

)

f1 = f1_score(

    true_labels,

    predictions

)

print("\n")

print("="*70)

print("Evaluation Metrics")

print("="*70)

print(

    "Accuracy :", round(

        accuracy,

        4

    )

)

print(

    "Precision :", round(

        precision,

        4

    )

)

print(

    "Recall :", round(

        recall,

        4

    )

)

print(

    "F1 Score :", round(

        f1,

        4

    )

)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    true_labels,

    predictions

)

plt.figure(

    figsize=(6,5)

)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.title(

    "Confusion Matrix"

)

plt.xlabel(

    "Predicted"

)

plt.ylabel(

    "Actual"

)

plt.show()

# ============================================================
# Classification Report
# ============================================================

print("\n")

print("="*70)

print("Classification Report")

print("="*70)

print(

    classification_report(

        true_labels,

        predictions,

        target_names=[

            "NORMAL",

            "PNEUMONIA"

        ]

    )

)

# ============================================================
# Training Accuracy
# ============================================================

plt.figure(

    figsize=(8,5)

)

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"

)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"

)

plt.legend()

plt.grid()

plt.title(

    "Training vs Validation Accuracy"

)

plt.show()

# ============================================================
# Training Loss
# ============================================================

plt.figure(

    figsize=(8,5)

)

plt.plot(

    history.history["loss"],

    label="Training Loss"

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"

)

plt.legend()

plt.grid()

plt.title(

    "Training vs Validation Loss"

)

plt.show()

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- ResNet50 uses residual learning to train deep networks.")

print("- Transfer Learning significantly reduces training time.")

print("- Freezing pretrained layers is ideal for CPU-based training.")

print("- Data augmentation improves model generalization.")

print("- Medical imaging benefits from pretrained ImageNet features.")

# ============================================================
# Final Model Summary
# ============================================================

print("\n")

print("="*70)

print("ResNet50 Model Summary")

print("="*70)

print("Architecture            : ResNet50")

print("Transfer Learning       : Yes")

print("Frozen Layers           : Yes")

print("Epochs                  : 5")

print("Image Size              : 128 x 128")

print("Batch Size              : 16")

print("Optimizer               : Adam")

print("Loss Function           : Binary Crossentropy")

print("Accuracy                :", round(accuracy,4))

print("Precision               :", round(precision,4))

print("Recall                  :", round(recall,4))

print("F1 Score                :", round(f1,4))

print("="*70)

# ============================================================
#  Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = load_model(

    "best_resnet50_model.keras"

)

print("\n")

print("="*70)

print("ResNet50 Model Loaded Successfully")

print("="*70)

# ============================================================
# Predict Single Image
# ============================================================

image_path = "chest_xray/test/NORMAL/NORMAL2-IM-0372-0001.jpeg"

img = image.load_img(

    image_path,

    target_size=(128,128)

)

img_array = image.img_to_array(

    img

)

img_array = img_array / 255.0

img_array = np.expand_dims(

    img_array,

    axis=0

)

prediction = loaded_model.predict(

    img_array,

    verbose=0

)

predicted_class = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

confidence = prediction[0][0]

print("\n")

print("="*70)

print("Prediction Result")

print("="*70)

print(

    "Predicted Class :",

    predicted_class

)

print(

    "Confidence Score :",

    round(

        float(confidence),

        4

    )

)

# ============================================================
# Display Image
# ============================================================

plt.figure(

    figsize=(5,5)

)

plt.imshow(

    img

)

plt.title(

    predicted_class

)

plt.axis("off")

plt.show()

# ============================================================
# Predict Multiple Images
# ============================================================

prediction_results = []

sample_folder = "chest_xray/test/NORMAL"

sample_images = os.listdir(

    sample_folder

)[:10]

for file in sample_images:

    path = os.path.join(

        sample_folder,

        file

    )

    img = image.load_img(

        path,

        target_size=(128,128)

    )

    img_array = image.img_to_array(

        img

    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(

        img_array,

        axis=0

    )

    pred = loaded_model.predict(

        img_array,

        verbose=0

    )

    label = "PNEUMONIA" if pred[0][0] > 0.5 else "NORMAL"

    prediction_results.append(

        [

            file,

            label,

            float(pred[0][0])

        ]

    )

prediction_df = pd.DataFrame(

    prediction_results,

    columns=[

        "Image",

        "Prediction",

        "Confidence"

    ]

)

print("\n")

print("="*70)

print("Prediction Results")

print("="*70)

display(

    prediction_df

)

# ============================================================
# Export Prediction Results
# ============================================================

prediction_df.to_csv(

    "Medical_Image_Predictions.csv",

    index=False

)

print("\n")

print("="*70)

print("Predictions Exported Successfully")

print("="*70)

# ============================================================
# Save Evaluation Metrics
# ============================================================

evaluation_metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ],

    "Value":[

        accuracy,

        precision,

        recall,

        f1

    ]

})

evaluation_metrics.to_csv(

    "Evaluation_Metrics.csv",

    index=False

)

print("\n")

print("="*70)

print("Evaluation Metrics Saved Successfully")

print("="*70)

display(

    evaluation_metrics

)

# ============================================================
# Training History
# ============================================================

history_df = pd.DataFrame(

    history.history

)

history_df.to_csv(

    "Training_History.csv",

    index=False

)

print("\n")

print("="*70)

print("Training History Saved Successfully")

print("="*70)

# ============================================================
# Deployment Summary
# ============================================================

deployment_summary = pd.DataFrame({

    "File":[

        "ResNet50 Model",

        "Prediction Results",

        "Evaluation Metrics",

        "Training History"

    ],

    "Saved As":[

        "best_resnet50_model.keras",

        "Medical_Image_Predictions.csv",

        "Evaluation_Metrics.csv",

        "Training_History.csv"

    ]

})

print("\n")

print("="*70)

print("Deployment Summary")

print("="*70)

display(

    deployment_summary

)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- Transfer Learning dramatically reduces training time.")

print("- Frozen pretrained layers allow efficient CPU training.")

print("- Medical image classification can assist radiologists.")

print("- AI predictions should support, not replace, clinical diagnosis.")

print("- Periodic retraining with updated datasets improves robustness.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Medical Imaging using ResNet50")

print("="*70)

print("Dataset                 : Chest X-Ray Pneumonia")

print("Architecture            : ResNet50")

print("Transfer Learning       : Yes")

print("Problem Type            : Binary Image Classification")

print("Classes                 : NORMAL / PNEUMONIA")

print("Image Size              : 128 x 128")

print("Batch Size              : 16")

print("Epochs                  : 5")

print("Optimizer               : Adam")

print("Loss Function           : Binary Crossentropy")

print("Accuracy                :", round(accuracy,4))

print("Precision               :", round(precision,4))

print("Recall                  :", round(recall,4))

print("F1 Score                :", round(f1,4))

print("Model Saved             : best_resnet50_model.keras")

print("Prediction File         : Medical_Image_Predictions.csv")

print("Metrics File            : Evaluation_Metrics.csv")

print("Training History File   : Training_History.csv")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Medical Imaging using ResNet50 Completed Successfully!")

print("="*70)


