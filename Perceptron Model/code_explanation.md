# ============================================================
# Basic Classification using Perceptron Model
# Part-1A : Data Loading & Exploratory Data Analysis
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.datasets import load_breast_cancer

warnings.filterwarnings("ignore")

# ============================================================
# Load Dataset
# ============================================================

cancer = load_breast_cancer()

df = pd.DataFrame(

    cancer.data,

    columns=cancer.feature_names

)

df["Target"] = cancer.target

print("\n")

print("="*70)

print("Dataset Loaded Successfully")

print("="*70)

# ============================================================
# Display First Five Records
# ============================================================

print("\n")

print("="*70)

print("First Five Records")

print("="*70)

display(

    df.head()

)

# ============================================================
# Display Last Five Records
# ============================================================

print("\n")

print("="*70)

print("Last Five Records")

print("="*70)

display(

    df.tail()

)

# ============================================================
# Dataset Shape
# ============================================================

print("\n")

print("="*70)

print("Dataset Shape")

print("="*70)

print(

    "Rows    :",

    df.shape[0]

)

print(

    "Columns :",

    df.shape[1]

)

# ============================================================
# Dataset Information
# ============================================================

print("\n")

print("="*70)

print("Dataset Information")

print("="*70)

df.info()

# ============================================================
# Column Names
# ============================================================

print("\n")

print("="*70)

print("Column Names")

print("="*70)

print(

    df.columns.tolist()

)

# ============================================================
# Data Types
# ============================================================

print("\n")

print("="*70)

print("Data Types")

print("="*70)

display(

    df.dtypes

)

# ============================================================
# Statistical Summary
# ============================================================

print("\n")

print("="*70)

print("Statistical Summary")

print("="*70)

display(

    df.describe()

)

# ============================================================
# Complete Statistical Summary
# ============================================================

print("\n")

print("="*70)

print("Complete Statistical Summary")

print("="*70)

display(

    df.describe(

        include="all"

    )

)

# ============================================================
# Missing Values
# ============================================================

print("\n")

print("="*70)

print("Missing Values")

print("="*70)

display(

    df.isnull().sum()

)

# ============================================================
# Duplicate Records
# ============================================================

duplicates = df.duplicated().sum()

print("\n")

print("="*70)

print("Duplicate Records")

print("="*70)

print(

    "Duplicate Records :",

    duplicates

)

# ============================================================
# Remove Duplicate Records
# ============================================================

df.drop_duplicates(

    inplace=True

)

print("\n")

print("="*70)

print("Duplicate Records Removed Successfully")

print("="*70)

print(

    "Current Shape :",

    df.shape

)

# ============================================================
# Target Class Distribution
# ============================================================

plt.figure(

    figsize=(6,5)

)

sns.countplot(

    x="Target",

    data=df

)

plt.title(

    "Target Class Distribution"

)

plt.xlabel(

    "Target"

)

plt.ylabel(

    "Count"

)

plt.show()

# ============================================================
# Class Percentage
# ============================================================

class_percentage = df["Target"].value_counts(

    normalize=True

) * 100

print("\n")

print("="*70)

print("Class Percentage")

print("="*70)

display(

    class_percentage

)

# ============================================================
# Feature Distribution
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.histplot(

    df["mean radius"],

    bins=30,

    kde=True

)

plt.title(

    "Mean Radius Distribution"

)

plt.show()

# ============================================================
# Boxplot
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.boxplot(

    x=df["mean area"]

)

plt.title(

    "Mean Area Boxplot"

)

plt.show()

# ============================================================
# Correlation Matrix
# ============================================================

plt.figure(

    figsize=(16,12)

)

sns.heatmap(

    df.corr(),

    cmap="coolwarm"

)

plt.title(

    "Correlation Matrix"

)

plt.show()

# ============================================================
# Top Correlated Features
# ============================================================

correlation = df.corr()["Target"].sort_values(

    ascending=False

)

print("\n")

print("="*70)

print("Top Correlated Features with Target")

print("="*70)

display(

    correlation.head(

        10

    )

)

# ============================================================
# Pairplot (Selected Features)
# ============================================================

selected_features = [

    "mean radius",

    "mean texture",

    "mean perimeter",

    "Target"

]

sns.pairplot(

    df[selected_features],

    hue="Target"

)

plt.show()

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

print(

    "Total Records      :",

    df.shape[0]

)

print(

    "Total Features     :",

    df.shape[1] - 1

)

print(

    "Target Variable    : Target"

)

print(

    "Class 0 (Malignant):",

    df["Target"].value_counts()[0]

)

print(

    "Class 1 (Benign)   :",

    df["Target"].value_counts()[1]

)

print(

    "Missing Values     :",

    df.isnull().sum().sum()

)

print(

    "Duplicate Records  :",

    duplicates

)

print("="*70)

# ============================================================
#  Data Preprocessing
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# ============================================================
# Check Missing Values
# ============================================================

print("\n")

print("="*70)

print("Missing Values Before Treatment")

print("="*70)

display(

    df.isnull().sum()

)

# ============================================================
# Handle Missing Values
# ============================================================

df.fillna(

    df.mean(

        numeric_only=True

    ),

    inplace=True

)

print("\n")

print("="*70)

print("Missing Values After Treatment")

print("="*70)

display(

    df.isnull().sum()

)

# ============================================================
# Check Duplicate Records
# ============================================================

duplicates = df.duplicated().sum()

print("\n")

print("="*70)

print("Duplicate Records")

print("="*70)

print(

    "Duplicate Records :",

    duplicates

)

# ============================================================
# Remove Duplicate Records
# ============================================================

df.drop_duplicates(

    inplace=True

)

print("\n")

print("="*70)

print("Duplicate Records Removed")

print("="*70)

print(

    "Current Shape :",

    df.shape

)

# ============================================================
# Feature Matrix
# ============================================================

X = df.drop(

    "Target",

    axis=1

)

# ============================================================
# Target Variable
# ============================================================

y = df["Target"]

print("\n")

print("="*70)

print("Feature Matrix & Target Variable")

print("="*70)

print(

    "Features Shape :",

    X.shape

)

print(

    "Target Shape :",

    y.shape

)

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\n")

print("="*70)

print("Train Test Split Completed")

print("="*70)

print(

    "Training Samples :",

    len(X_train)

)

print(

    "Testing Samples :",

    len(X_test)

)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(

    X_train

)

X_test = scaler.transform(

    X_test

)

print("\n")

print("="*70)

print("Feature Scaling Completed")

print("="*70)

# ============================================================
# Convert Back to DataFrame
# ============================================================

X_train = pd.DataFrame(

    X_train,

    columns=X.columns

)

X_test = pd.DataFrame(

    X_test,

    columns=X.columns

)

print("\n")

print("="*70)

print("Scaled Training Dataset")

print("="*70)

display(

    X_train.head()

)

print("\n")

print("="*70)

print("Scaled Testing Dataset")

print("="*70)

display(

    X_test.head()

)

# ============================================================
# Target Distribution
# ============================================================

print("\n")

print("="*70)

print("Training Target Distribution")

print("="*70)

display(

    y_train.value_counts()

)

print("\n")

print("="*70)

print("Testing Target Distribution")

print("="*70)

display(

    y_test.value_counts()

)

# ============================================================
# Feature Statistics After Scaling
# ============================================================

print("\n")

print("="*70)

print("Scaled Feature Statistics")

print("="*70)

display(

    X_train.describe()

)

# ============================================================
# Verify Scaling
# ============================================================

print("\n")

print("="*70)

print("Feature Scaling Verification")

print("="*70)

print(

    "Mean (Approx.) :",

    round(

        X_train.mean().mean(),

        4

    )

)

print(

    "Std (Approx.) :",

    round(

        X_train.std().mean(),

        4

    )

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

print(

    "Training Samples :",

    X_train.shape[0]

)

print(

    "Testing Samples :",

    X_test.shape[0]

)

print(

    "Number of Features :",

    X_train.shape[1]

)

print(

    "Number of Classes :",

    len(

        np.unique(

            y

        )

    )

)

print("="*70)

# ============================================================
# Dataset Ready
# ============================================================

print("\n")

print("="*70)

print("Dataset Ready for Perceptron Model")

print("="*70)

# ============================================================
#  Perceptron Model Building & Evaluation
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.linear_model import Perceptron

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import seaborn as sns

# ============================================================
# Build Perceptron Model
# ============================================================

model = Perceptron(

    max_iter=1000,

    eta0=0.01,

    random_state=42,

    tol=1e-3

)

print("\n")

print("="*70)

print("Perceptron Model Created Successfully")

print("="*70)

# ============================================================
# Train Model
# ============================================================

model.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)

print("Model Training Completed")

print("="*70)

# ============================================================
# Model Parameters
# ============================================================

print("\n")

print("="*70)

print("Model Parameters")

print("="*70)

print(

    "Epochs Used :",

    model.n_iter_

)

print(

    "Classes :",

    model.classes_

)

# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(

    X_test

)

print("\n")

print("="*70)

print("Predictions Generated")

print("="*70)

display(

    pd.DataFrame({

        "Actual":y_test.values,

        "Predicted":y_pred

    }).head(20)

)

# ============================================================
# Evaluation Metrics
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

print("\n")

print("="*70)

print("Model Evaluation")

print("="*70)

print(

    "Accuracy  :",

    round(accuracy,4)

)

print(

    "Precision :",

    round(precision,4)

)

print(

    "Recall    :",

    round(recall,4)

)

print(

    "F1 Score  :",

    round(f1,4)

)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred

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

        y_test,

        y_pred

    )

)

# ============================================================
# Model Weights
# ============================================================

weights = pd.DataFrame({

    "Feature":X_train.columns,

    "Weight":model.coef_[0]

})

weights["Absolute Weight"] = weights["Weight"].abs()

weights = weights.sort_values(

    by="Absolute Weight",

    ascending=False

)

print("\n")

print("="*70)

print("Top 10 Important Features")

print("="*70)

display(

    weights.head(10)

)

# ============================================================
# Feature Weight Visualization
# ============================================================

plt.figure(

    figsize=(10,6)

)

sns.barplot(

    data=weights.head(10),

    x="Weight",

    y="Feature"

)

plt.title(

    "Top 10 Feature Weights"

)

plt.show()

# ============================================================
# Learning Information
# ============================================================

print("\n")

print("="*70)

print("Learning Information")

print("="*70)

print(

    "Learning Rate :",

    model.eta0

)

print(

    "Maximum Epochs :",

    model.max_iter

)

print(

    "Bias (Intercept) :",

    round(

        model.intercept_[0],

        4

    )

)

# ============================================================
# Decision Boundary Visualization
# (Using only first two features)
# ============================================================

from matplotlib.colors import ListedColormap

X_visual = X_train.iloc[:,0:2]

y_visual = y_train

visual_model = Perceptron(

    max_iter=1000,

    random_state=42

)

visual_model.fit(

    X_visual,

    y_visual

)

x1_min = X_visual.iloc[:,0].min()-1

x1_max = X_visual.iloc[:,0].max()+1

x2_min = X_visual.iloc[:,1].min()-1

x2_max = X_visual.iloc[:,1].max()+1

xx1, xx2 = np.meshgrid(

    np.arange(

        x1_min,

        x1_max,

        0.02

    ),

    np.arange(

        x2_min,

        x2_max,

        0.02

    )

)

prediction = visual_model.predict(

    np.array(

        [

            xx1.ravel(),

            xx2.ravel()

        ]

    ).T

)

prediction = prediction.reshape(

    xx1.shape

)

plt.figure(

    figsize=(8,6)

)

plt.contourf(

    xx1,

    xx2,

    prediction,

    alpha=0.3,

    cmap=ListedColormap(

        ("red","green")

    )

)

plt.scatter(

    X_visual.iloc[:,0],

    X_visual.iloc[:,1],

    c=y_visual,

    cmap=ListedColormap(

        ("red","green")

    )

)

plt.title(

    "Perceptron Decision Boundary"

)

plt.xlabel(

    X_train.columns[0]

)

plt.ylabel(

    X_train.columns[1]

)

plt.show()

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- Perceptron is the foundation of Artificial Neural Networks.")

print("- It works well only when the data is linearly separable.")

print("- Feature scaling significantly improves convergence.")

print("- Multiple perceptrons form hidden layers in deep learning.")

print("- Modern neural networks extend the perceptron concept with multiple layers.")

# ============================================================
# Final Model Summary
# ============================================================

print("\n")

print("="*70)

print("Perceptron Model Summary")

print("="*70)

print("Algorithm               : Perceptron")

print("Problem Type            : Binary Classification")

print("Training Samples        :",len(X_train))

print("Testing Samples         :",len(X_test))

print("Features                :",X_train.shape[1])

print("Epochs Used             :",model.n_iter_)

print("Accuracy                :",round(accuracy,4))

print("Precision               :",round(precision,4))

print("Recall                  :",round(recall,4))

print("F1 Score                :",round(f1,4))

print("="*70)

# ============================================================
#  Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

# ============================================================
# Save Trained Model
# ============================================================

joblib.dump(

    model,

    "perceptron_model.joblib"

)

print("\n")

print("="*70)

print("Perceptron Model Saved Successfully")

print("="*70)

# ============================================================
# Save Scaler
# ============================================================

joblib.dump(

    scaler,

    "standard_scaler.joblib"

)

print("\n")

print("="*70)

print("Standard Scaler Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = joblib.load(

    "perceptron_model.joblib"

)

loaded_scaler = joblib.load(

    "standard_scaler.joblib"

)

print("\n")

print("="*70)

print("Saved Model Loaded Successfully")

print("="*70)

# ============================================================
# Predict Test Dataset
# ============================================================

loaded_predictions = loaded_model.predict(

    X_test

)

prediction_output = pd.DataFrame({

    "Actual":y_test.values,

    "Predicted":loaded_predictions

})

print("\n")

print("="*70)

print("Prediction Sample")

print("="*70)

display(

    prediction_output.head(

        20

    )

)

# ============================================================
# Export Predictions
# ============================================================

prediction_output.to_csv(

    "Perceptron_Predictions.csv",

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
# Export Feature Weights
# ============================================================

weights.to_csv(

    "Feature_Weights.csv",

    index=False

)

print("\n")

print("="*70)

print("Feature Weights Exported Successfully")

print("="*70)

display(

    weights.head(

        10

    )

)

# ============================================================
# Feature Weight Visualization
# ============================================================

plt.figure(

    figsize=(10,6)

)

sns.barplot(

    data=weights.head(10),

    x="Weight",

    y="Feature"

)

plt.title(

    "Top 10 Feature Weights"

)

plt.xlabel(

    "Weight"

)

plt.ylabel(

    "Feature"

)

plt.show()

# ============================================================
# Predict New Sample
# ============================================================

new_sample = X_test.iloc[[0]]

prediction = loaded_model.predict(

    new_sample

)

prediction_label = "Benign" if prediction[0] == 1 else "Malignant"

print("\n")

print("="*70)

print("New Sample Prediction")

print("="*70)

print(

    "Predicted Class :",

    prediction_label

)

# ============================================================
# Model Information
# ============================================================

model_information = pd.DataFrame({

    "Parameter":[

        "Algorithm",

        "Model Type",

        "Training Samples",

        "Testing Samples",

        "Epochs",

        "Learning Rate"

    ],

    "Value":[

        "Perceptron",

        "Binary Classification",

        len(X_train),

        len(X_test),

        model.n_iter_,

        model.eta0

    ]

})

print("\n")

print("="*70)

print("Model Information")

print("="*70)

display(

    model_information

)

# ============================================================
# Deployment Summary
# ============================================================

deployment_summary = pd.DataFrame({

    "File":[

        "Perceptron Model",

        "Standard Scaler",

        "Prediction Results",

        "Evaluation Metrics",

        "Feature Weights"

    ],

    "Saved As":[

        "perceptron_model.joblib",

        "standard_scaler.joblib",

        "Perceptron_Predictions.csv",

        "Evaluation_Metrics.csv",

        "Feature_Weights.csv"

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

print("- Perceptron is the first neural network model.")

print("- Feature scaling improves convergence speed.")

print("- The model learns a linear decision boundary.")

print("- Perceptron works well for linearly separable datasets.")

print("- Multi-layer neural networks overcome Perceptron limitations.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Basic Classification using Perceptron")

print("="*70)

print("Dataset                 : Breast Cancer Wisconsin")

print("Algorithm               : Perceptron")

print("Problem Type            : Binary Classification")

print("Training Samples        :",len(X_train))

print("Testing Samples         :",len(X_test))

print("Features                :",X_train.shape[1])

print("Epochs                  :",model.n_iter_)

print("Learning Rate           :",model.eta0)

print("Accuracy                :",round(accuracy,4))

print("Precision               :",round(precision,4))

print("Recall                  :",round(recall,4))

print("F1 Score                :",round(f1,4))

print("Model Saved             : perceptron_model.joblib")

print("Scaler Saved            : standard_scaler.joblib")

print("Prediction File         : Perceptron_Predictions.csv")

print("Feature Weights File    : Feature_Weights.csv")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Basic Classification using Perceptron Completed Successfully!")

print("="*70)

