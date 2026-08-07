# ============================================================
# Image Classification using EfficientNetB0
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

dataset_path = "seg_train"

print("\n")

print("="*70)

print("Dataset Path Loaded Successfully")

print("="*70)

# ============================================================
# Class Names
# ============================================================

classes = sorted(

    os.listdir(dataset_path)

)

print("\n")

print("="*70)

print("Classes")

print("="*70)

print(classes)

# ============================================================
# Count Images
# ============================================================

image_count = {}

for category in classes:

    folder = os.path.join(

        dataset_path,

        category

    )

    image_count[category] = len(

        os.listdir(folder)

    )

count_df = pd.DataFrame({

    "Class":image_count.keys(),

    "Images":image_count.values()

})

print("\n")

print("="*70)

print("Images Per Class")

print("="*70)

display(

    count_df

)

# ============================================================
# Total Images
# ============================================================

total_images = sum(

    image_count.values()

)

print("\n")

print("="*70)

print("Total Images")

print("="*70)

print(total_images)

# ============================================================
# Class Distribution
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.barplot(

    data=count_df,

    x="Class",

    y="Images"

)

plt.xticks(

    rotation=20

)

plt.title(

    "Class Distribution"

)

plt.show()

# ============================================================
# Display Sample Images
# ============================================================

plt.figure(

    figsize=(15,8)

)

for i, category in enumerate(classes):

    image_name = os.listdir(

        os.path.join(

            dataset_path,

            category

        )

    )[0]

    image_path = os.path.join(

        dataset_path,

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

        2,

        3,

        i+1

    )

    plt.imshow(

        image

    )

    plt.title(

        category

    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# ============================================================
# Image Resolution Analysis
# ============================================================

heights = []

widths = []

for category in classes:

    folder = os.path.join(

        dataset_path,

        category

    )

    image_files = os.listdir(folder)[:100]

    for file in image_files:

        img = cv2.imread(

            os.path.join(

                folder,

                file

            )

        )

        if img is not None:

            h, w, _ = img.shape

            heights.append(h)

            widths.append(w)

print("\n")

print("="*70)

print("Resolution Statistics")

print("="*70)

print(

    "Average Height :",

    round(

        np.mean(

            heights

        ),

        2

    )

)

print(

    "Average Width :",

    round(

        np.mean(

            widths

        ),

        2

    )

)

print(

    "Minimum Height :",

    min(

        heights

    )

)

print(

    "Maximum Height :",

    max(

        heights

    )

)

print(

    "Minimum Width :",

    min(

        widths

    )

)

print(

    "Maximum Width :",

    max(

        widths

    )

)

# ============================================================
# Pixel Distribution
# ============================================================

gray = cv2.imread(

    image_path,

    cv2.IMREAD_GRAYSCALE

)

plt.figure(

    figsize=(8,5)

)

plt.hist(

    gray.ravel(),

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
# Sample Image Shape
# ============================================================

print("\n")

print("="*70)

print("Sample Image Shape")

print("="*70)

print(

    gray.shape

)

# ============================================================
# Image Channels
# ============================================================

rgb = cv2.imread(

    image_path

)

print("\n")

print("="*70)

print("RGB Image Shape")

print("="*70)

print(

    rgb.shape

)

# ============================================================
# Dataset Information
# ============================================================

dataset_info = pd.DataFrame({

    "Parameter":[

        "Dataset",

        "Classes",

        "Total Images",

        "Input Image Size",

        "Recommended Model"

    ],

    "Value":[

        "Intel Image Classification",

        len(classes),

        total_images,

        "160 x 160",

        "EfficientNetB0"

    ]

})

print("\n")

print("="*70)

print("Dataset Information")

print("="*70)

display(

    dataset_info

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

print(

    "Total Classes :", len(classes)

)

print(

    "Total Images :", total_images

)

print(

    "Average Height :", round(

        np.mean(

            heights

        ),

        2

    )

)

print(

    "Average Width :", round(

        np.mean(

            widths

        ),

        2

    )

)

print(

    "Recommended Image Size : 160 x 160"

)

print(

    "Transfer Learning Model : EfficientNetB0"

)

print("="*70)



print("="*70)

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
# Dataset Paths
# ============================================================

train_path = "seg_train"

test_path = "seg_test"

# ============================================================
# Image Configuration (CPU Optimized)
# ============================================================

IMAGE_SIZE = (160,160)

BATCH_SIZE = 16

print("\n")

print("="*70)

print("Image Configuration")

print("="*70)

print("Image Size :", IMAGE_SIZE)

print("Batch Size :", BATCH_SIZE)

# ============================================================
# Training Data Generator
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.20,

    rotation_range=10,

    zoom_range=0.10,

    width_shift_range=0.10,

    height_shift_range=0.10,

    horizontal_flip=True

)

# ============================================================
# Test Data Generator
# ============================================================

test_datagen = ImageDataGenerator(

    rescale=1./255

)

# ============================================================
# Training Generator
# ============================================================

train_generator = train_datagen.flow_from_directory(

    train_path,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True

)

# ============================================================
# Validation Generator
# ============================================================

validation_generator = train_datagen.flow_from_directory(

    train_path,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False

)

# ============================================================
# Testing Generator
# ============================================================

test_generator = test_datagen.flow_from_directory(

    test_path,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

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
# Display One Batch
# ============================================================

images, labels = next(

    train_generator

)

print("\n")

print("="*70)

print("Batch Shape")

print("="*70)

print(

    "Images :", images.shape

)

print(

    "Labels :", labels.shape

)

# ============================================================
# Display Sample Images
# ============================================================

class_names = list(

    train_generator.class_indices.keys()

)

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

        class_names[

            np.argmax(

                labels[i]

            )

        ]

    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# ============================================================
# Pixel Verification
# ============================================================

print("\n")

print("="*70)

print("Pixel Value Verification")

print("="*70)

print(

    "Minimum Pixel :", np.min(images)

)

print(

    "Maximum Pixel :", np.max(images)

)

# ============================================================
# Dataset Information
# ============================================================

dataset_information = pd.DataFrame({

    "Parameter":[

        "Training Images",

        "Validation Images",

        "Testing Images",

        "Number of Classes",

        "Image Size",

        "Batch Size"

    ],

    "Value":[

        train_generator.samples,

        validation_generator.samples,

        test_generator.samples,

        train_generator.num_classes,

        "160 x 160",

        BATCH_SIZE

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
# Training Configuration
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

    "Validation Steps :", validation_steps

)

print(

    "Test Steps :", test_steps

)

# ============================================================
# Dataset Ready
# ============================================================

print("\n")

print("="*70)

print("Dataset Ready for EfficientNetB0")

print("="*70)

# ============================================================
#  EfficientNetB0 Model Building & Evaluation
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

# ============================================================
# Load EfficientNetB0
# ============================================================

base_model = EfficientNetB0(

    weights="imagenet",

    include_top=False,

    input_shape=(160,160,3)

)

print("\n")

print("="*70)

print("EfficientNetB0 Loaded Successfully")

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

    256,

    activation="relu"

)(x)

x = Dropout(

    0.30

)(x)

output = Dense(

    train_generator.num_classes,

    activation="softmax"

)(x)

model = Model(

    inputs=base_model.input,

    outputs=output

)

print("\n")

print("="*70)

print("Classification Head Added")

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

    loss="categorical_crossentropy",

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

    "best_efficientnet_model.keras",

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

print(round(accuracy,4))

# ============================================================
# Predictions
# ============================================================

prediction_probability = model.predict(

    test_generator,

    verbose=0

)

predictions = np.argmax(

    prediction_probability,

    axis=1

)

true_labels = test_generator.classes

# ============================================================
# Evaluation Metrics
# ============================================================

precision = precision_score(

    true_labels,

    predictions,

    average="weighted"

)

recall = recall_score(

    true_labels,

    predictions,

    average="weighted"

)

f1 = f1_score(

    true_labels,

    predictions,

    average="weighted"

)

print("\n")

print("="*70)

print("Evaluation Metrics")

print("="*70)

print("Accuracy  :", round(accuracy,4))

print("Precision :", round(precision,4))

print("Recall    :", round(recall,4))

print("F1 Score  :", round(f1,4))

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    true_labels,

    predictions

)

plt.figure(

    figsize=(8,6)

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

        target_names=list(

            train_generator.class_indices.keys()

        )

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

print("- EfficientNetB0 uses compound scaling to balance depth, width, and resolution.")

print("- Transfer learning significantly reduces training time.")

print("- Frozen pretrained layers make CPU training much faster.")

print("- Data augmentation improves model generalization.")

print("- EfficientNetB0 achieves high accuracy with fewer parameters than many older CNNs.")

# ============================================================
# Final Model Summary
# ============================================================

print("\n")

print("="*70)

print("EfficientNetB0 Model Summary")

print("="*70)

print("Architecture            : EfficientNetB0")

print("Transfer Learning       : Yes")

print("Frozen Layers           : Yes")

print("Epochs                  : 5")

print("Image Size              : 160 x 160")

print("Batch Size              : 16")

print("Optimizer               : Adam")

print("Loss Function           : Categorical Crossentropy")

print("Classes                 :", train_generator.num_classes)

print("Accuracy                :", round(accuracy,4))

print("Precision               :", round(precision,4))

print("Recall                  :", round(recall,4))

print("F1 Score                :", round(f1,4))

print("="*70)

# ============================================================
# Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import os

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

    "best_efficientnet_model.keras"

)

print("\n")

print("="*70)

print("EfficientNetB0 Model Loaded Successfully")

print("="*70)

# ============================================================
# Class Names
# ============================================================

class_names = list(

    train_generator.class_indices.keys()

)

print("\n")

print("="*70)

print("Class Names")

print("="*70)

print(class_names)

# ============================================================
# Predict Single Image
# ============================================================

image_path = "seg_test/buildings/20057.jpg"

img = image.load_img(

    image_path,

    target_size=(160,160)

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

predicted_index = np.argmax(

    prediction

)

predicted_class = class_names[

    predicted_index

]

confidence = np.max(

    prediction

)

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

sample_folder = "seg_test/buildings"

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

        target_size=(160,160)

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

    label = class_names[

        np.argmax(

            pred

        )

    ]

    confidence = np.max(

        pred

    )

    prediction_results.append(

        [

            file,

            label,

            float(

                confidence

            )

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

    "EfficientNet_Predictions.csv",

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
# Save Training History
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

        "EfficientNet Model",

        "Prediction Results",

        "Evaluation Metrics",

        "Training History"

    ],

    "Saved As":[

        "best_efficientnet_model.keras",

        "EfficientNet_Predictions.csv",

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

print("- EfficientNetB0 achieves excellent accuracy with relatively few parameters.")

print("- Compound scaling improves efficiency by balancing network depth, width, and image resolution.")

print("- Transfer learning enables high performance with limited training time.")

print("- Freezing pretrained layers makes EfficientNet practical even on CPUs.")

print("- EfficientNetB0 is suitable for real-world image classification applications.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Image Classification using EfficientNetB0")

print("="*70)

print("Dataset                 : Intel Image Classification")

print("Architecture            : EfficientNetB0")

print("Transfer Learning       : Yes")

print("Problem Type            : Multi-Class Image Classification")

print("Number of Classes       :", train_generator.num_classes)

print("Image Size              : 160 x 160")

print("Batch Size              : 16")

print("Epochs                  : 5")

print("Optimizer               : Adam")

print("Loss Function           : Categorical Crossentropy")

print("Accuracy                :", round(accuracy,4))

print("Precision               :", round(precision,4))

print("Recall                  :", round(recall,4))

print("F1 Score                :", round(f1,4))

print("Model Saved             : best_efficientnet_model.keras")

print("Prediction File         : EfficientNet_Predictions.csv")

print("Metrics File            : Evaluation_Metrics.csv")

print("Training History File   : Training_History.csv")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Image Classification using EfficientNetB0 Completed Successfully!")

print("="*70)


