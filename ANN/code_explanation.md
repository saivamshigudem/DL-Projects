# ============================================================
# PART-1: EXPLORATORY DATA ANALYSIS (EDA)
#        + DATA PREPROCESSING
# ============================================================


# ============================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ============================================================

# CODE EXPLANATION:
# pandas      -> Used for loading and manipulating datasets
# numpy       -> Used for numerical operations
# matplotlib  -> Used for visualization
# seaborn     -> Used for advanced visualization
# train_test_split -> Splits dataset into training and testing data
# LabelEncoder -> Converts categorical text values into numbers
# StandardScaler -> Normalizes feature values for ANN


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)



# ============================================================
# SECTION 2: LOAD DATASET
# ============================================================

# CODE EXPLANATION:
# Loads the customer churn dataset from CSV file.
#
# Each row represents one customer.
# Each column represents customer information.
#
# Dataset:
# Telco Customer Churn Dataset


df = pd.read_csv(
    "customer_churn.csv"
)



# ============================================================
# SECTION 3: DISPLAY FIRST FIVE RECORDS
# ============================================================

# CODE EXPLANATION:
# head() displays first five rows.
#
# Used to verify:
# - Dataset loaded successfully
# - Column names
# - Data format


df.head()



# ============================================================
# SECTION 4: CHECK DATASET SIZE
# ============================================================

# CODE EXPLANATION:
# shape returns:
#
# Number of rows
# Number of columns
#
# Example:
# (7043,21)
#
# Means:
# 7043 customer records
# 21 columns


df.shape



# ============================================================
# SECTION 5: DATASET INFORMATION
# ============================================================

# CODE EXPLANATION:
# info() displays:
#
# Column names
# Data types
# Missing values
#
# Helps identify:
# - Numerical columns
# - Categorical columns


df.info()



# ============================================================
# SECTION 6: STATISTICAL ANALYSIS
# ============================================================

# CODE EXPLANATION:
# describe() provides numerical statistics:
#
# Mean
# Standard deviation
# Minimum
# Maximum
# Quartiles


df.describe()



# ============================================================
# SECTION 7: CHECK MISSING VALUES
# ============================================================

# CODE EXPLANATION:
# Finds missing values in every column.
#
# Missing values need to be handled
# before ANN training.


df.isnull().sum()



# ============================================================
# SECTION 8: REMOVE CUSTOMER ID COLUMN
# ============================================================

# CODE EXPLANATION:
#
# customerID is only an identifier.
#
# It does not contain useful customer behaviour information.
#
# Removing it improves model learning.


df.drop("CustomerID", axis=1, inplace=True, errors="ignore")


# ============================================================
# SECTION 9: CONVERT TOTALCHARGES TO NUMERIC
# ============================================================

# CODE EXPLANATION:
#
# TotalCharges contains empty string values.
#
# Convert it into numerical format.
#
# errors="coerce":
# Invalid values become NaN.


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)



# ============================================================
# SECTION 10: REMOVE MISSING RECORDS
# ============================================================

# CODE EXPLANATION:
#
# Removes rows containing missing values.
#
# Keeps only clean customer records.


df.dropna(
    inplace=True
)



# ============================================================
# SECTION 11: CHURN DISTRIBUTION VISUALIZATION
# ============================================================

# CODE EXPLANATION:
#
# Shows number of customers:
#
# Who stayed
# Who left
#
# Helps understand target balance.


sns.countplot(
    x="Churn",
    data=df
)


plt.title(
    "Customer Churn Distribution"
)


plt.show()



# ============================================================
# SECTION 12: CONVERT TARGET COLUMN
# ============================================================

# CODE EXPLANATION:
#
# ANN requires numerical output.
#
# Convert:
#
# Yes = 1
# No  = 0


df["Churn"] = df["Churn"].map(
    {
        "Yes":1,
        "No":0
    }
)



# ============================================================
# SECTION 13: ENCODE CATEGORICAL FEATURES
# ============================================================

# CODE EXPLANATION:
#
# Neural networks cannot process text.
#
# Example:
#
# InternetService
#
# DSL
# Fiber
# No
#
# Converted into:
#
# 0
# 1
# 2


encoder = LabelEncoder()


for column in df.columns:

    if df[column].dtype == "object":

        df[column] = encoder.fit_transform(
            df[column]
        )



# ============================================================
# SECTION 14: VERIFY ENCODED DATA
# ============================================================

# CODE EXPLANATION:
#
# Checks whether all values
# are converted into numerical format.


df.head()



# ============================================================
# SECTION 15: SPLIT FEATURES AND TARGET
# ============================================================

# CODE EXPLANATION:
#
# X:
# Independent variables/features
#
# y:
# Dependent variable/target
#
# Model learns relationship between X and y.


X = df.drop(
    "Churn",
    axis=1
)


y = df["Churn"]



# ============================================================
# SECTION 16: TRAIN TEST SPLIT
# ============================================================

# CODE EXPLANATION:
#
# Dataset split:
#
# 80% -> Training
# 20% -> Testing
#
# Training data:
# Used for learning patterns
#
# Testing data:
# Used for final evaluation


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)



# ============================================================
# SECTION 17: FEATURE SCALING
# ============================================================

# CODE EXPLANATION:
#
# ANN works better when input features
# are in similar ranges.
#
# StandardScaler converts values into:
#
# Mean = 0
# Standard deviation = 1
#
# Important:
# Fit scaler only on training data.


scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_test = scaler.transform(
    X_test
)



# ============================================================
# SECTION 18: FINAL DATA VERIFICATION
# ============================================================

# CODE EXPLANATION:
#
# Checks final shape before ANN training.
#
# Output:
# Number of samples
# Number of features


print(
    "Training Data Shape:",
    X_train.shape
)


print(
    "Testing Data Shape:",
    X_test.shape
)


print(
    "Target Training Shape:",
    y_train.shape
)


print(
    "Target Testing Shape:",
    y_test.shape
)

# ============================================================
# ANN MODEL DEVELOPMENT + TRAINING + EVALUATION
# ============================================================


# ============================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ============================================================

# CODE EXPLANATION:
#
# TensorFlow is used to build the Artificial Neural Network.
#
# Sequential:
# Creates a neural network layer by layer.
#
# Dense:
# Creates a fully connected layer.
#
# Dropout:
# Prevents overfitting by randomly disabling neurons.
#
# EarlyStopping:
# Stops training automatically when validation performance
# no longer improves.
#
# sklearn.metrics:
# Used for evaluating the model.


import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)



# ============================================================
# SECTION 2: CREATE ANN MODEL
# ============================================================

# CODE EXPLANATION:
#
# Sequential creates a feed-forward neural network.
#
# Network Architecture:
#
# Input Layer
#        │
#        ▼
# Hidden Layer (32 neurons)
#        │
#        ▼
# Dropout (20%)
#        │
#        ▼
# Hidden Layer (16 neurons)
#        │
#        ▼
# Output Layer (1 neuron)


model = Sequential()



# ============================================================
# SECTION 3: INPUT LAYER + FIRST HIDDEN LAYER
# ============================================================

# CODE EXPLANATION:
#
# Dense:
# Fully connected layer.
#
# 32:
# Number of neurons.
#
# activation="relu":
# Uses Rectified Linear Unit.
#
# Formula:
#
# f(x) = max(0,x)
#
# input_shape:
# Number of input features.


model.add(

    Dense(

        units=32,

        activation="relu",

        input_shape=(X_train.shape[1],)

    )

)



# ============================================================
# SECTION 4: DROPOUT LAYER
# ============================================================

# CODE EXPLANATION:
#
# Randomly ignores 20% neurons during training.
#
# Advantages:
#
# Prevents overfitting
#
# Improves generalization
#
# Makes model more robust


model.add(

    Dropout(

        rate=0.20

    )

)



# ============================================================
# SECTION 5: SECOND HIDDEN LAYER
# ============================================================

# CODE EXPLANATION:
#
# Smaller hidden layer.
#
# Learns higher-level patterns.
#
# CPU-friendly architecture.


model.add(

    Dense(

        units=16,

        activation="relu"

    )

)



# ============================================================
# SECTION 6: OUTPUT LAYER
# ============================================================

# CODE EXPLANATION:
#
# Binary Classification
#
# Output neuron = 1
#
# Sigmoid activation returns probability
# between 0 and 1.
#
# Example:
#
# 0.91
#
# Means:
# 91% chance customer will churn.


model.add(

    Dense(

        units=1,

        activation="sigmoid"

    )

)



# ============================================================
# SECTION 7: DISPLAY MODEL ARCHITECTURE
# ============================================================

# CODE EXPLANATION:
#
# Displays:
#
# Layer names
#
# Output shape
#
# Number of trainable parameters


model.summary()



# ============================================================
# SECTION 8: COMPILE MODEL
# ============================================================

# CODE EXPLANATION:
#
# optimizer="adam"
#
# Adam automatically updates weights.
#
# loss="binary_crossentropy"
#
# Suitable for binary classification.
#
# metrics=["accuracy"]
#
# Tracks prediction accuracy.


model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)



# ============================================================
# SECTION 9: EARLY STOPPING
# ============================================================

# CODE EXPLANATION:
#
# CPU Optimization
#
# If validation loss does not improve
# for 5 consecutive epochs,
# stop training automatically.
#
# restore_best_weights=True
#
# Restores the best-performing model.


early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)



# ============================================================
# SECTION 10: TRAIN ANN MODEL
# ============================================================

# CODE EXPLANATION:
#
# epochs=50
#
# Maximum number of training iterations.
#
# validation_split=0.2
#
# Uses 20% of training data for validation.
#
# batch_size=32
#
# Processes 32 samples at once.
#
# callbacks
#
# Enables EarlyStopping.


history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=50,

    batch_size=32,

    callbacks=[early_stop],

    verbose=1

)



# ============================================================
# SECTION 11: PLOT TRAINING ACCURACY
# ============================================================

# CODE EXPLANATION:
#
# Shows how accuracy changes
# during training.


plt.figure(figsize=(8,5))

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"

)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"

)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()



# ============================================================
# SECTION 12: PLOT TRAINING LOSS
# ============================================================

# CODE EXPLANATION:
#
# Lower loss indicates
# better learning.


plt.figure(figsize=(8,5))

plt.plot(

    history.history["loss"],

    label="Training Loss"

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"

)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()



# ============================================================
# SECTION 13: MAKE PREDICTIONS
# ============================================================

# CODE EXPLANATION:
#
# predict()
#
# Returns probabilities.
#
# Example:
#
# 0.84
#
# Convert probabilities into
# class labels.
#
# Threshold:
#
# >=0.5 → Churn
#
# <0.5 → No Churn


prediction_probability = model.predict(X_test)

prediction = (

    prediction_probability >= 0.5

).astype(int)



# ============================================================
# SECTION 14: CALCULATE ACCURACY
# ============================================================

# CODE EXPLANATION:
#
# Accuracy =
#
# Correct Predictions
# -------------------
# Total Predictions


accuracy = accuracy_score(

    y_test,

    prediction

)

print(f"Model Accuracy : {accuracy:.4f}")



# ============================================================
# SECTION 15: CONFUSION MATRIX
# ============================================================

# CODE EXPLANATION:
#
# Shows:
#
# True Positive
#
# True Negative
#
# False Positive
#
# False Negative


cm = confusion_matrix(

    y_test,

    prediction

)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()



# ============================================================
# SECTION 16: CLASSIFICATION REPORT
# ============================================================

# CODE EXPLANATION:
#
# Displays:
#
# Precision
#
# Recall
#
# F1 Score
#
# Support


print(

    classification_report(

        y_test,

        prediction

    )

)



# ============================================================
# SECTION 17: ROC-AUC SCORE
# ============================================================

# CODE EXPLANATION:
#
# ROC-AUC measures the model's ability
# to distinguish between classes.
#
# Closer to 1.0 is better.


auc = roc_auc_score(

    y_test,

    prediction_probability

)

print(f"ROC-AUC Score : {auc:.4f}")



# ============================================================
# SECTION 18: ROC CURVE
# ============================================================

# CODE EXPLANATION:
#
# Plots the Receiver Operating Characteristic (ROC) curve.
#
# The closer the curve follows the top-left corner,
# the better the model.


fpr, tpr, thresholds = roc_curve(

    y_test,

    prediction_probability

)

plt.figure(figsize=(7,6))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {auc:.3f}"

)

plt.plot(

    [0,1],

    [0,1],

    linestyle="--"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()



# ============================================================
# SECTION 19: SAVE TRAINED MODEL
# ============================================================

# CODE EXPLANATION:
#
# Saves the trained ANN model.
#
# This file will be used
# during deployment.


model.save("customer_churn_ann.keras")

print("Model Saved Successfully!")


# ============================================================
# MODEL DEPLOYMENT USING FASTAPI
# ============================================================


# ============================================================
# SECTION 1: PROJECT STRUCTURE
# ============================================================

# CODE EXPLANATION:
#
# Organize the project in the following structure.
#
# Customer_Churn_ANN/
#
# │
# ├── customer_churn.csv
# ├── customer_churn_ann.keras
# ├── scaler.pkl
# ├── app.py
# ├── requirements.txt
# └── README.md
#
# This structure is easy to deploy on
# Render
# Railway
# AWS EC2
# Azure
# Docker


# ============================================================
# SECTION 2: SAVE THE SCALER
# ============================================================

# CODE EXPLANATION:
#
# During deployment we must preprocess
# incoming customer data exactly like
# training data.
#
# Therefore we save the fitted scaler.


import joblib

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Scaler Saved Successfully")
