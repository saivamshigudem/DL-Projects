# ============================================================
# Object Detection using YOLOv8
# : Dataset Exploration & EDA
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import os

import cv2

import yaml

import random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Dataset Path
# ============================================================

dataset_path = "Fruit_Detection"

train_images = os.path.join(

    dataset_path,

    "train",

    "images"

)

train_labels = os.path.join(

    dataset_path,

    "train",

    "labels"

)

valid_images = os.path.join(

    dataset_path,

    "valid",

    "images"

)

test_images = os.path.join(

    dataset_path,

    "test",

    "images"

)

yaml_file = os.path.join(

    dataset_path,

    "data.yaml"

)

print("\n")

print("="*70)

print("Dataset Path Loaded Successfully")

print("="*70)

# ============================================================
# Read data.yaml
# ============================================================

with open(

    yaml_file,

    "r"

) as file:

    data = yaml.safe_load(file)

print("\n")

print("="*70)

print("Dataset Information")

print("="*70)

display(data)

# ============================================================
# Class Names
# ============================================================

class_names = data["names"]

print("\n")

print("="*70)

print("Class Names")

print("="*70)

print(class_names)

# ============================================================
# Number of Classes
# ============================================================

print("\n")

print("="*70)

print("Number of Classes")

print("="*70)

print(

    len(class_names)

)

# ============================================================
# Count Images
# ============================================================

train_count = len(

    os.listdir(train_images)

)

valid_count = len(

    os.listdir(valid_images)

)

test_count = len(

    os.listdir(test_images)

)

print("\n")

print("="*70)

print("Dataset Size")

print("="*70)

print(

    "Training Images :", train_count

)

print(

    "Validation Images :", valid_count

)

print(

    "Testing Images :", test_count

)

# ============================================================
# Dataset Summary
# ============================================================

summary = pd.DataFrame({

    "Dataset":[

        "Train",

        "Validation",

        "Test"

    ],

    "Images":[

        train_count,

        valid_count,

        test_count

    ]

})

display(summary)

# ============================================================
# Display Sample Images
# ============================================================

sample_images = random.sample(

    os.listdir(train_images),

    6

)

plt.figure(

    figsize=(15,8)

)

for i, file in enumerate(sample_images):

    image = cv2.imread(

        os.path.join(

            train_images,

            file

        )

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

    plt.title(file)

    plt.axis("off")

plt.tight_layout()

plt.show()

# ============================================================
# Image Resolution Analysis
# ============================================================

heights = []

widths = []

image_files = os.listdir(

    train_images

)[:100]

for file in image_files:

    img = cv2.imread(

        os.path.join(

            train_images,

            file

        )

    )

    if img is not None:

        h, w, _ = img.shape

        heights.append(h)

        widths.append(w)

print("\n")

print("="*70)

print("Image Resolution")

print("="*70)

print(

    "Average Height :",

    round(

        np.mean(heights),

        2

    )

)

print(

    "Average Width :",

    round(

        np.mean(widths),

        2

    )

)

print(

    "Minimum Height :",

    min(heights)

)

print(

    "Maximum Height :",

    max(heights)

)

print(

    "Minimum Width :",

    min(widths)

)

print(

    "Maximum Width :",

    max(widths)

)

# ============================================================
# Read Label Files
# ============================================================

label_files = os.listdir(

    train_labels

)

object_counter = {}

for cls in class_names:

    object_counter[cls] = 0

for label in label_files:

    with open(

        os.path.join(

            train_labels,

            label

        ),

        "r"

    ) as file:

        lines = file.readlines()

        for line in lines:

            cls_id = int(

                line.split()[0]

            )

            object_counter[

                class_names[cls_id]

            ] += 1

# ============================================================
# Object Distribution
# ============================================================

object_df = pd.DataFrame({

    "Class":object_counter.keys(),

    "Objects":object_counter.values()

})

print("\n")

print("="*70)

print("Object Distribution")

print("="*70)

display(object_df)

plt.figure(

    figsize=(8,5)

)

sns.barplot(

    data=object_df,

    x="Class",

    y="Objects"

)

plt.title(

    "Object Distribution"

)

plt.xticks(

    rotation=30

)

plt.show()

# ============================================================
# Sample Label File
# ============================================================

sample_label = label_files[0]

print("\n")

print("="*70)

print("Sample YOLO Label")

print("="*70)

with open(

    os.path.join(

        train_labels,

        sample_label

    ),

    "r"

) as file:

    print(

        file.read()

    )

# ============================================================
# Dataset Information
# ============================================================

dataset_information = pd.DataFrame({

    "Parameter":[

        "Dataset",

        "Model",

        "Classes",

        "Training Images",

        "Validation Images",

        "Testing Images"

    ],

    "Value":[

        "Fruit Detection",

        "YOLOv8",

        len(class_names),

        train_count,

        valid_count,

        test_count

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
#  Dataset Preparation & Validation
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import os

import yaml

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import numpy as np

# ============================================================
# Verify Dataset Structure
# ============================================================

required_folders = [

    train_images,

    train_labels,

    valid_images,

    test_images

]

print("\n")

print("="*70)

print("Dataset Folder Verification")

print("="*70)

for folder in required_folders:

    if os.path.exists(folder):

        print("✓", folder)

    else:

        print("✗", folder)

# ============================================================
# Verify data.yaml
# ============================================================

print("\n")

print("="*70)

print("Reading data.yaml")

print("="*70)

with open(

    yaml_file,

    "r"

) as file:

    dataset_yaml = yaml.safe_load(file)

display(

    dataset_yaml

)

# ============================================================
# Dataset Configuration
# ============================================================

print("\n")

print("="*70)

print("Dataset Configuration")

print("="*70)

print(

    "Training Path :",

    dataset_yaml["train"]

)

print(

    "Validation Path :",

    dataset_yaml["val"]

)

print(

    "Number of Classes :",

    dataset_yaml["nc"]

)

print(

    "Class Names :",

    dataset_yaml["names"]

)

# ============================================================
# Count Label Files
# ============================================================

train_label_count = len(

    os.listdir(train_labels)

)

print("\n")

print("="*70)

print("Label Files")

print("="*70)

print(

    "Training Labels :",

    train_label_count

)

# ============================================================
# Check Image-Label Pair
# ============================================================

missing_labels = 0

image_files = os.listdir(

    train_images

)

for img in image_files:

    txt = os.path.splitext(

        img

    )[0] + ".txt"

    if not os.path.exists(

        os.path.join(

            train_labels,

            txt

        )

    ):

        missing_labels += 1

print("\n")

print("="*70)

print("Image-Label Verification")

print("="*70)

print(

    "Missing Labels :",

    missing_labels

)

# ============================================================
# Display One Image with Bounding Boxes
# ============================================================

sample_image = image_files[0]

image_path = os.path.join(

    train_images,

    sample_image

)

label_path = os.path.join(

    train_labels,

    os.path.splitext(

        sample_image

    )[0] + ".txt"

)

img = cv2.imread(

    image_path

)

img = cv2.cvtColor(

    img,

    cv2.COLOR_BGR2RGB

)

height, width, _ = img.shape

with open(

    label_path,

    "r"

) as file:

    lines = file.readlines()

for line in lines:

    cls, xc, yc, w, h = map(

        float,

        line.split()

    )

    xc *= width

    yc *= height

    w *= width

    h *= height

    x1 = int(

        xc - w/2

    )

    y1 = int(

        yc - h/2

    )

    x2 = int(

        xc + w/2

    )

    y2 = int(

        yc + h/2

    )

    cv2.rectangle(

        img,

        (x1,y1),

        (x2,y2),

        (255,0,0),

        2

    )

    cv2.putText(

        img,

        class_names[int(cls)],

        (x1,y1-5),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255,0,0),

        2

    )

plt.figure(

    figsize=(8,8)

)

plt.imshow(

    img

)

plt.title(

    "Bounding Box Visualization"

)

plt.axis("off")

plt.show()

# ============================================================
# Training Configuration
# ============================================================

IMAGE_SIZE = 416

BATCH_SIZE = 8

EPOCHS = 10

MODEL_NAME = "yolov8n.pt"

print("\n")

print("="*70)

print("Training Configuration")

print("="*70)

print(

    "Model :", MODEL_NAME

)

print(

    "Image Size :", IMAGE_SIZE

)

print(

    "Batch Size :", BATCH_SIZE

)

print(

    "Epochs :", EPOCHS

)

# ============================================================
# CPU Optimization
# ============================================================

cpu_settings = pd.DataFrame({

    "Setting":[

        "Model",

        "Image Size",

        "Batch Size",

        "Epochs",

        "Workers",

        "Device"

    ],

    "Value":[

        "YOLOv8 Nano",

        "416 x 416",

        8,

        10,

        2,

        "CPU"

    ]

})

print("\n")

print("="*70)

print("CPU Optimized Settings")

print("="*70)

display(

    cpu_settings

)

# ============================================================
# Dataset Summary
# ============================================================

dataset_summary = pd.DataFrame({

    "Parameter":[

        "Dataset",

        "Model",

        "Classes",

        "Training Images",

        "Validation Images",

        "Testing Images"

    ],

    "Value":[

        "Fruit Detection",

        "YOLOv8 Nano",

        len(class_names),

        train_count,

        valid_count,

        test_count

    ]

})

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

display(

    dataset_summary

)

# ============================================================
# Dataset Ready
# ============================================================

print("\n")

print("="*70)

print("Dataset Ready for YOLOv8 Training")

print("="*70)

# ============================================================
#  Model Training & Evaluation
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

from ultralytics import YOLO

import matplotlib.pyplot as plt

import pandas as pd

# ============================================================
# Load Pretrained YOLOv8 Nano Model
# ============================================================

model = YOLO(

    "yolov8n.pt"

)

print("\n")

print("="*70)

print("YOLOv8 Nano Model Loaded Successfully")

print("="*70)

# ============================================================
# Train Model
# ============================================================

results = model.train(

    data=yaml_file,

    epochs=10,

    imgsz=416,

    batch=8,

    workers=2,

    device="cpu",

    cache=False,

    project="YOLO_Project",

    name="Fruit_Detection",

    verbose=True

)

print("\n")

print("="*70)

print("Model Training Completed")

print("="*70)

# ============================================================
# Validate Model
# ============================================================

metrics = model.val()

print("\n")

print("="*70)

print("Validation Completed")

print("="*70)

# ============================================================
# Performance Metrics
# ============================================================

print("\n")

print("="*70)

print("Performance Metrics")

print("="*70)

print(

    "Precision :", round(

        float(metrics.box.mp),

        4

    )

)

print(

    "Recall :", round(

        float(metrics.box.mr),

        4

    )

)

print(

    "mAP@0.5 :", round(

        float(metrics.box.map50),

        4

    )

)

print(

    "mAP@0.5:0.95 :", round(

        float(metrics.box.map),

        4

    )

)

# ============================================================
# Predict Sample Images
# ============================================================

prediction = model.predict(

    source=test_images,

    imgsz=416,

    conf=0.25,

    device="cpu",

    save=True

)

print("\n")

print("="*70)

print("Prediction Completed")

print("="*70)

# ============================================================
# Display Prediction Image
# ============================================================

prediction_image = prediction[0].plot()

plt.figure(

    figsize=(8,8)

)

plt.imshow(

    prediction_image

)

plt.axis("off")

plt.title(

    "YOLOv8 Prediction"

)

plt.show()

# ============================================================
# Metrics Summary
# ============================================================

metrics_df = pd.DataFrame({

    "Metric":[

        "Precision",

        "Recall",

        "mAP@0.5",

        "mAP@0.5:0.95"

    ],

    "Value":[

        float(metrics.box.mp),

        float(metrics.box.mr),

        float(metrics.box.map50),

        float(metrics.box.map)

    ]

})

print("\n")

print("="*70)

print("Evaluation Summary")

print("="*70)

display(

    metrics_df

)

# ============================================================
# Training Results
# ============================================================

results_csv = "YOLO_Project/Fruit_Detection/results.csv"

if os.path.exists(results_csv):

    history = pd.read_csv(

        results_csv

    )

    print("\n")

    print("="*70)

    print("Training History")

    print("="*70)

    display(

        history.head()

    )

# ============================================================
# Plot Training Loss
# ============================================================

if os.path.exists(results_csv):

    plt.figure(

        figsize=(8,5)

    )

    plt.plot(

        history["train/box_loss"],

        label="Box Loss"

    )

    plt.plot(

        history["val/box_loss"],

        label="Validation Box Loss"

    )

    plt.legend()

    plt.grid()

    plt.title(

        "Box Loss"

    )

    plt.show()

# ============================================================
# Plot mAP
# ============================================================

if os.path.exists(results_csv):

    plt.figure(

        figsize=(8,5)

    )

    plt.plot(

        history["metrics/mAP50(B)"],

        label="mAP@0.5"

    )

    plt.plot(

        history["metrics/mAP50-95(B)"],

        label="mAP@0.5:0.95"

    )

    plt.legend()

    plt.grid()

    plt.title(

        "Detection Accuracy"

    )

    plt.show()

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- YOLO predicts object class and location simultaneously.")

print("- Transfer Learning reduces training time.")

print("- mAP is the most important metric for object detection.")

print("- Precision measures prediction correctness.")

print("- Recall measures how many objects are detected.")

print("- YOLOv8 Nano is ideal for CPU deployment.")

# ============================================================
# Final Model Summary
# ============================================================

print("\n")

print("="*70)

print("YOLOv8 Model Summary")

print("="*70)

print("Architecture           : YOLOv8 Nano")

print("Transfer Learning      : Yes")

print("Problem Type           : Object Detection")

print("Image Size             : 416 x 416")

print("Batch Size             : 8")

print("Epochs                 : 10")

print("Device                 : CPU")

print("Precision              :", round(float(metrics.box.mp),4))

print("Recall                 :", round(float(metrics.box.mr),4))

print("mAP@0.5                :", round(float(metrics.box.map50),4))

print("mAP@0.5:0.95           :", round(float(metrics.box.map),4))

print("="*70)

# ============================================================
#  Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import os

import cv2

import pandas as pd

import matplotlib.pyplot as plt

from ultralytics import YOLO

# ============================================================
# Load Best Trained Model
# ============================================================

best_model = YOLO(

    "YOLO_Project/Fruit_Detection/weights/best.pt"

)

print("\n")

print("="*70)

print("Best YOLOv8 Model Loaded Successfully")

print("="*70)




# ============================================================
# Save Evaluation Metrics
# ============================================================

evaluation_metrics = pd.DataFrame({

    "Metric":[

        "Precision",

        "Recall",

        "mAP@0.5",

        "mAP@0.5:0.95"

    ],

    "Value":[

        float(metrics.box.mp),

        float(metrics.box.mr),

        float(metrics.box.map50),

        float(metrics.box.map)

    ]

})

evaluation_metrics.to_csv(

    "YOLO_Evaluation_Metrics.csv",

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
# Deployment Summary
# ============================================================

deployment_summary = pd.DataFrame({

    "File":[

        "Best YOLO Model",

        "Detection Results",

        "Evaluation Metrics",

        "Training History"

    ],

    "Saved As":[

        "best.pt",

        "YOLO_Detection_Results.csv",

        "YOLO_Evaluation_Metrics.csv",

        "YOLO_Training_History.csv"

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

print("- YOLO detects multiple objects in a single forward pass.")

print("- Bounding boxes help localize object positions.")

print("- Confidence score represents prediction certainty.")

print("- mAP is the primary evaluation metric for object detection.")

print("- YOLOv8 Nano provides an excellent balance between speed and accuracy.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Object Detection using YOLOv8")

print("="*70)

print("Dataset                 : Fruit Detection")

print("Architecture            : YOLOv8 Nano")

print("Problem Type            : Object Detection")

print("Transfer Learning       : Yes")

print("Image Size              : 416 x 416")

print("Batch Size              : 8")

print("Epochs                  : 10")

print("Device                  : CPU")

print("Precision               :", round(float(metrics.box.mp),4))

print("Recall                  :", round(float(metrics.box.mr),4))

print("mAP@0.5                 :", round(float(metrics.box.map50),4))

print("mAP@0.5:0.95            :", round(float(metrics.box.map),4))

print("Model Saved             : best.pt")

print("Detection Results       : YOLO_Detection_Results.csv")

print("Metrics File            : YOLO_Evaluation_Metrics.csv")

print("Training History File   : YOLO_Training_History.csv")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Object Detection using YOLOv8 Completed Successfully!")

print("="*70)



