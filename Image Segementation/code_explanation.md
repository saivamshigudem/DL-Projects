# ============================================================
# Medical Image Segmentation using U-Net
# Part-1A : Data Loading & Exploratory Data Analysis
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import os

import cv2

import random

import warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# Dataset Paths
# ============================================================

IMAGE_PATH = "ISBI2016_ISIC_Part1_Training_Data"

MASK_PATH = "ISBI2016_ISIC_Part1_Training_GroundTruth"

print("\n")
print("="*70)
print("Dataset Loaded Successfully")
print("="*70)

# ============================================================
# Read Image Names
# ============================================================

image_files = sorted(

    os.listdir(

        IMAGE_PATH

    )

)

mask_files = sorted(

    os.listdir(

        MASK_PATH

    )

)

print("\n")
print("="*70)
print("Dataset Information")
print("="*70)

print(

    "Images :", len(image_files)

)

print(

    "Masks :", len(mask_files)

)

# ============================================================
# Dataset Summary
# ============================================================

summary = pd.DataFrame({

    "Parameter":[

        "Dataset",

        "Task",

        "Images",

        "Masks"

    ],

    "Value":[

        "ISIC 2016",

        "Skin Lesion Segmentation",

        len(image_files),

        len(mask_files)

    ]

})

display(summary)

# ============================================================
# Verify Dataset
# ============================================================

print("\n")
print("="*70)
print("Dataset Verification")
print("="*70)

if len(image_files) == len(mask_files):

    print("✓ Images and Masks Matched")

else:

    print("✗ Images and Masks Count Mismatch")

# ============================================================
# Display Sample Images
# ============================================================

sample_indices = random.sample(

    range(

        len(image_files)

    ),

    4

)

plt.figure(

    figsize=(12,8)

)

for i, idx in enumerate(sample_indices):

    img = cv2.imread(

        os.path.join(

            IMAGE_PATH,

            image_files[idx]

        )

    )

    img = cv2.cvtColor(

        img,

        cv2.COLOR_BGR2RGB

    )

    plt.subplot(

        2,

        2,

        i+1

    )

    plt.imshow(

        img

    )

    plt.title(

        image_files[idx]

    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# ============================================================
# Display Image and Corresponding Mask
# ============================================================

sample_image = image_files[0]

mask_name = sample_image.replace(

    ".jpg",

    "_Segmentation.png"

)

image = cv2.imread(

    os.path.join(

        IMAGE_PATH,

        sample_image

    )

)

image = cv2.cvtColor(

    image,

    cv2.COLOR_BGR2RGB

)

mask = cv2.imread(

    os.path.join(

        MASK_PATH,

        mask_name

    ),

    cv2.IMREAD_GRAYSCALE

)

plt.figure(

    figsize=(10,5)

)

plt.subplot(

    1,

    2,

    1

)

plt.imshow(

    image

)

plt.title("Original Image")

plt.axis("off")

plt.subplot(

    1,

    2,

    2

)

plt.imshow(

    mask,

    cmap="gray"

)

plt.title("Ground Truth Mask")

plt.axis("off")

plt.show()

# ============================================================
# Image Resolution Analysis
# ============================================================

heights = []

widths = []

for file in image_files[:100]:

    img = cv2.imread(

        os.path.join(

            IMAGE_PATH,

            file

        )

    )

    h, w, _ = img.shape

    heights.append(h)

    widths.append(w)

print("\n")
print("="*70)
print("Image Resolution Statistics")
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

plt.figure(

    figsize=(7,5)

)

plt.hist(

    image.ravel(),

    bins=50

)

plt.title(

    "Pixel Distribution"

)

plt.xlabel(

    "Pixel Value"

)

plt.ylabel(

    "Frequency"

)

plt.show()

# ============================================================
# Mask Pixel Distribution
# ============================================================

unique_pixels = np.unique(

    mask

)

print("\n")
print("="*70)
print("Mask Pixel Values")
print("="*70)

print(

    unique_pixels

)

# ============================================================
# Image Shape
# ============================================================

print("\n")
print("="*70)
print("Image Shape")
print("="*70)

print(

    image.shape

)

print(

    mask.shape

)

# ============================================================
# Dataset Information
# ============================================================

dataset_info = pd.DataFrame({

    "Parameter":[

        "Model",

        "Problem Type",

        "Classes",

        "Recommended Image Size",

        "Epochs",

        "Batch Size"

    ],

    "Value":[

        "Lightweight U-Net",

        "Binary Segmentation",

        2,

        "128 x 128",

        10,

        8

    ]

})

display(dataset_info)
# ============================================================
#  Image & Mask Preprocessing
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import os

import cv2

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# ============================================================
# Configuration (CPU Optimized)
# ============================================================

IMAGE_SIZE = 128

BATCH_SIZE = 8

print("\n")
print("="*70)
print("Configuration")
print("="*70)
print("Image Size :", IMAGE_SIZE)
print("Batch Size :", BATCH_SIZE)

# ============================================================
# Read Images & Masks
# ============================================================

images = []

masks = []

for image_name in sorted(os.listdir(IMAGE_PATH)):

    image_path = os.path.join(

        IMAGE_PATH,

        image_name

    )

    mask_name = image_name.replace(

        ".jpg",

        "_Segmentation.png"

    )

    mask_path = os.path.join(

        MASK_PATH,

        mask_name

    )

    if not os.path.exists(mask_path):

        continue

    # Image

    image = cv2.imread(image_path)

    image = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )

    image = cv2.resize(

        image,

        (IMAGE_SIZE, IMAGE_SIZE)

    )

    image = image.astype(

        np.float32

    ) / 255.0

    # Mask

    mask = cv2.imread(

        mask_path,

        cv2.IMREAD_GRAYSCALE

    )

    mask = cv2.resize(

        mask,

        (IMAGE_SIZE, IMAGE_SIZE),

        interpolation=cv2.INTER_NEAREST

    )

    mask = mask.astype(

        np.float32

    ) / 255.0

    mask = np.expand_dims(

        mask,

        axis=-1

    )

    images.append(image)

    masks.append(mask)

# ============================================================
# Convert to NumPy Arrays
# ============================================================

images = np.array(

    images,

    dtype=np.float32

)

masks = np.array(

    masks,

    dtype=np.float32

)

print("\n")
print("="*70)
print("Dataset Loaded")
print("="*70)
print("Images :", images.shape)
print("Masks  :", masks.shape)

# ============================================================
# Train / Validation Split
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(

    images,

    masks,

    test_size=0.20,

    random_state=42,

    shuffle=True

)

print("\n")
print("="*70)
print("Dataset Split")
print("="*70)

print("Training Images :", X_train.shape)

print("Validation Images :", X_valid.shape)

# ============================================================
# Verify Pixel Range
# ============================================================

print("\n")
print("="*70)
print("Pixel Verification")
print("="*70)

print("Image Min :", X_train.min())

print("Image Max :", X_train.max())

print("Mask Min :", y_train.min())

print("Mask Max :", y_train.max())

# ============================================================
# Display Sample Image & Mask
# ============================================================

index = 0

plt.figure(

    figsize=(10,5)

)

plt.subplot(

    1,

    2,

    1

)

plt.imshow(

    X_train[index]

)

plt.title(

    "Training Image"

)

plt.axis("off")

plt.subplot(

    1,

    2,

    2

)

plt.imshow(

    y_train[index].squeeze(),

    cmap="gray"

)

plt.title(

    "Training Mask"

)

plt.axis("off")

plt.show()

# ============================================================
# Dataset Information
# ============================================================

print("\n")
print("="*70)
print("Dataset Information")
print("="*70)

print("Training Samples :", len(X_train))

print("Validation Samples :", len(X_valid))

print("Image Shape :", X_train.shape[1:])

print("Mask Shape :", y_train.shape[1:])

print("Batch Size :", BATCH_SIZE)

print("Image Size :", IMAGE_SIZE)

# ============================================================
# Create TensorFlow Dataset
# ============================================================

import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices(

    (X_train, y_train)

)

train_dataset = train_dataset.batch(

    BATCH_SIZE

).prefetch(

    tf.data.AUTOTUNE

)

validation_dataset = tf.data.Dataset.from_tensor_slices(

    (X_valid, y_valid)

)

validation_dataset = validation_dataset.batch(

    BATCH_SIZE

).prefetch(

    tf.data.AUTOTUNE

)

print("\n")
print("="*70)
print("TensorFlow Dataset Created Successfully")
print("="*70)

# ============================================================
# Dataset Ready
# ============================================================

print("\n")
print("="*70)
print("Dataset Ready for Lightweight U-Net")
print("="*70)

# ============================================================

#  Lightweight U-Net
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import tensorflow as tf

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Conv2DTranspose

from tensorflow.keras.layers import Concatenate

from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

import numpy as np

# ============================================================
# Dice Coefficient
# ============================================================

def dice_coefficient(y_true, y_pred):

    smooth = 1e-6

    y_true = tf.keras.backend.flatten(y_true)

    y_pred = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(y_true * y_pred)

    return (

        2. * intersection + smooth

    ) / (

        tf.reduce_sum(y_true)

        + tf.reduce_sum(y_pred)

        + smooth

    )

# ============================================================
# IoU Metric
# ============================================================

def iou_score(y_true, y_pred):

    y_pred = tf.cast(

        y_pred > 0.5,

        tf.float32

    )

    intersection = tf.reduce_sum(

        y_true * y_pred

    )

    union = tf.reduce_sum(

        y_true

    ) + tf.reduce_sum(

        y_pred

    ) - intersection

    return (

        intersection + 1e-6

    ) / (

        union + 1e-6

    )

# ============================================================
# U-Net Block
# ============================================================
def conv_block(inputs, filters):

    x = Conv2D(

        filters,

        3,

        padding="same",

        activation="relu"

    )(inputs)

    x = Conv2D(

        filters,

        3,

        padding="same",

        activation="relu"

    )(x)

    return x




# ============================================================
# Build Lightweight U-Net
# ============================================================

inputs = Input(

    shape=(128,128,3)

)

# Encoder

c1 = conv_block(inputs,16)

p1 = MaxPooling2D()(c1)

c2 = conv_block(p1,32)

p2 = MaxPooling2D()(c2)

c3 = conv_block(p2,64)

p3 = MaxPooling2D()(c3)

# Bottleneck

b1 = conv_block(p3,128)

b1 = Dropout(0.3)(b1)

# Decoder

u1 = Conv2DTranspose(

    64,

    2,

    strides=2,

    padding="same"

)(b1)

u1 = Concatenate()([u1,c3])

c4 = conv_block(u1,64)

u2 = Conv2DTranspose(

    32,

    2,

    strides=2,

    padding="same"

)(c4)

u2 = Concatenate()([u2,c2])

c5 = conv_block(u2,32)

u3 = Conv2DTranspose(

    16,

    2,

    strides=2,

    padding="same"

)(c5)

u3 = Concatenate()([u3,c1])

c6 = conv_block(u3,16)

outputs = Conv2D(

    1,

    1,

    activation="sigmoid"

)(c6)

model = Model(

    inputs,

    outputs

)

print("\n")

print("="*70)

print("Lightweight U-Net Built Successfully")

print("="*70)

# ============================================================
# Compile Model
# ============================================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        dice_coefficient,

        iou_score

    ]

)

print("\n")

print("="*70)

print("Model Compiled Successfully")

print("="*70)

model.summary()

# ============================================================
# Callbacks
# ============================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "best_unet_model.keras",

    save_best_only=True,

    monitor="val_loss"

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    verbose=1

)

# ============================================================
# Train Model
# ============================================================

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=10,

    callbacks=[

        early_stop,

        checkpoint,

        reduce_lr

    ],

    verbose=1

)

print("\n")

print("="*70)

print("Training Completed")

print("="*70)

# ============================================================
# Evaluate Model
# ============================================================

results = model.evaluate(

    validation_dataset,

    verbose=0

)

print("\n")

print("="*70)

print("Evaluation Results")

print("="*70)

print("Loss :", round(results[0],4))

print("Accuracy :", round(results[1],4))

print("Dice Coefficient :", round(results[2],4))

print("IoU Score :", round(results[3],4))

# ============================================================
# Predict Sample Image
# ============================================================

prediction = model.predict(

    X_valid[:1],

    verbose=0

)

prediction = (

    prediction > 0.5

).astype(np.uint8)

# ============================================================
# Display Results
# ============================================================

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)

plt.imshow(X_valid[0])

plt.title("Original")

plt.axis("off")

plt.subplot(1,3,2)

plt.imshow(y_valid[0].squeeze(), cmap="gray")

plt.title("Ground Truth")

plt.axis("off")

plt.subplot(1,3,3)

plt.imshow(prediction[0].squeeze(), cmap="gray")

plt.title("Prediction")

plt.axis("off")

plt.show()

# ============================================================
# Training Accuracy
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train")

plt.plot(history.history["val_accuracy"], label="Validation")

plt.legend()

plt.grid()

plt.title("Accuracy")

plt.show()

# ============================================================
# Training Loss
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train")

plt.plot(history.history["val_loss"], label="Validation")

plt.legend()

plt.grid()

plt.title("Loss")

plt.show()

# ============================================================
# Dice Score
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["dice_coefficient"], label="Train Dice")

plt.plot(history.history["val_dice_coefficient"], label="Validation Dice")

plt.legend()

plt.grid()

plt.title("Dice Coefficient")

plt.show()

# ============================================================
# IoU Score
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["iou_score"], label="Train IoU")

plt.plot(history.history["val_iou_score"], label="Validation IoU")

plt.legend()

plt.grid()

plt.title("IoU Score")

plt.show()

# ============================================================
# Business Insights
# ============================================================

print("\n")
print("="*70)
print("Business Insights")
print("="*70)

print("- U-Net performs pixel-wise segmentation.")

print("- Skip connections preserve spatial information.")

print("- Dice Score is the primary metric in medical segmentation.")

print("- IoU measures overlap between prediction and ground truth.")

print("- Lightweight U-Net is suitable for CPU training.")

# ============================================================
# Final Summary
# ============================================================

print("\n")
print("="*70)
print("Lightweight U-Net Summary")
print("="*70)

print("Model : Lightweight U-Net")

print("Task : Binary Image Segmentation")

print("Image Size : 128 x 128")

print("Batch Size : 8")

print("Epochs : 10")

print("Optimizer : Adam")

print("Loss : Binary Crossentropy")

print("Accuracy :", round(results[1],4))

print("Dice :", round(results[2],4))

print("IoU :", round(results[3],4))

print("="*70)

print("\n")

# ============================================================
#  Deployment
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import os

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import load_model

# ============================================================
# Custom Metrics (Required to Load Model)
# ============================================================

def dice_coefficient(y_true, y_pred):

    smooth = 1e-6

    y_true = tf.keras.backend.flatten(y_true)

    y_pred = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(y_true * y_pred)

    return (2. * intersection + smooth) / (

        tf.reduce_sum(y_true) +

        tf.reduce_sum(y_pred) +

        smooth

    )

def iou_score(y_true, y_pred):

    y_pred = tf.cast(

        y_pred > 0.5,

        tf.float32

    )

    intersection = tf.reduce_sum(

        y_true * y_pred

    )

    union = (

        tf.reduce_sum(y_true)

        + tf.reduce_sum(y_pred)

        - intersection

    )

    return (

        intersection + 1e-6

    ) / (

        union + 1e-6

    )

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = load_model(

    "best_unet_model.keras",

    custom_objects={

        "dice_coefficient":dice_coefficient,

        "iou_score":iou_score

    }

)

print("\n")

print("="*70)

print("Lightweight U-Net Loaded Successfully")

print("="*70)

# ============================================================
# Predict Single Image
# ============================================================

sample_image = X_valid[0]

prediction = loaded_model.predict(

    np.expand_dims(

        sample_image,

        axis=0

    ),

    verbose=0

)

prediction = (

    prediction > 0.5

).astype(np.uint8)

# ============================================================
# Display Prediction
# ============================================================

plt.figure(

    figsize=(15,5)

)

plt.subplot(

    1,

    3,

    1

)

plt.imshow(

    sample_image

)

plt.title("Original")

plt.axis("off")

plt.subplot(

    1,

    3,

    2

)

plt.imshow(

    y_valid[0].squeeze(),

    cmap="gray"

)

plt.title("Ground Truth")

plt.axis("off")

plt.subplot(

    1,

    3,

    3

)

plt.imshow(

    prediction[0].squeeze(),

    cmap="gray"

)

plt.title("Predicted Mask")

plt.axis("off")

plt.show()

# ============================================================
# Overlay Prediction
# ============================================================

overlay = sample_image.copy()

mask = prediction[0].squeeze()

overlay[mask == 1] = [1,0,0]

plt.figure(

    figsize=(6,6)

)

plt.imshow(

    overlay

)

plt.title(

    "Segmentation Overlay"

)

plt.axis("off")

plt.show()

# ============================================================
# Predict Multiple Images
# ============================================================

prediction_results = []

for i in range(

    min(

        10,

        len(X_valid)

    )

):

    pred = loaded_model.predict(

        np.expand_dims(

            X_valid[i],

            axis=0

        ),

        verbose=0

    )

    pred = (

        pred > 0.5

    ).astype(np.uint8)

    lesion_pixels = np.sum(pred)

    prediction_results.append({

        "Image No":i+1,

        "Lesion Pixels":int(

            lesion_pixels

        )

    })

prediction_df = pd.DataFrame(

    prediction_results

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

    "Segmentation_Predictions.csv",

    index=False

)

print("\n")

print("="*70)

print("Prediction Results Saved")

print("="*70)

# ============================================================
# Save Evaluation Metrics
# ============================================================

evaluation_metrics = pd.DataFrame({

    "Metric":[

        "Loss",

        "Accuracy",

        "Dice Score",

        "IoU"

    ],

    "Value":[

        results[0],

        results[1],

        results[2],

        results[3]

    ]

})

evaluation_metrics.to_csv(

    "Evaluation_Metrics.csv",

    index=False

)

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

print("Training History Saved")

print("="*70)

# ============================================================
# Deployment Summary
# ============================================================

deployment = pd.DataFrame({

    "File":[

        "U-Net Model",

        "Predictions",

        "Evaluation",

        "Training History"

    ],

    "Saved As":[

        "best_unet_model.keras",

        "Segmentation_Predictions.csv",

        "Evaluation_Metrics.csv",

        "Training_History.csv"

    ]

})

print("\n")

print("="*70)

print("Deployment Summary")

print("="*70)

display(

    deployment

)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- U-Net predicts lesion boundaries pixel-by-pixel.")

print("- Dice Score is the primary evaluation metric.")

print("- IoU measures segmentation overlap.")

print("- Skip connections preserve fine image details.")

print("- Lightweight U-Net enables efficient CPU training.")

print("- Segmentation assists clinicians in identifying lesion regions.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Medical Image Segmentation using Lightweight U-Net")

print("="*70)

print("Dataset                 : ISIC 2016")

print("Problem Type            : Binary Segmentation")

print("Architecture            : Lightweight U-Net")

print("Image Size              : 128 x 128")

print("Batch Size              : 8")

print("Epochs                  : 10")

print("Optimizer               : Adam")

print("Loss Function           : Binary Crossentropy")

print("Accuracy                :", round(results[1],4))

print("Dice Score              :", round(results[2],4))

print("IoU Score               :", round(results[3],4))

print("Model Saved             : best_unet_model.keras")

print("Prediction File         : Segmentation_Predictions.csv")

print("Metrics File            : Evaluation_Metrics.csv")

print("Training History File   : Training_History.csv")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Medical Image Segmentation using Lightweight U-Net")

print("Completed Successfully!")

print("="*70)
