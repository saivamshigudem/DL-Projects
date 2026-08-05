# ============================================================
#  DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS
# PROJECT : Research Analysis using PyTorch
# DATASET : Graduate Admission Prediction
# ============================================================


# ============================================================
# SECTION 1 : IMPORT LIBRARIES
# ============================================================

# CODE EXPLANATION:
# Import all required libraries.
#
# pandas       -> Data Manipulation
# numpy        -> Numerical Operations
# matplotlib   -> Basic Visualization
# seaborn      -> Statistical Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# SECTION 2 : LOAD DATASET
# ============================================================

# CODE EXPLANATION:
# Read the CSV dataset into a pandas DataFrame.

df = pd.read_csv("Admission_Predict.csv")


# ============================================================
# SECTION 3 : DISPLAY DATASET
# ============================================================

# CODE EXPLANATION:
# Display first and last five records.

print("First Five Records")
print(df.head())

print("\nLast Five Records")
print(df.tail())


# ============================================================
# SECTION 4 : DATASET SHAPE
# ============================================================

# CODE EXPLANATION:
# Returns number of rows and columns.

print("\nDataset Shape")
print(df.shape)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# SECTION 5 : DATASET INFORMATION
# ============================================================

# CODE EXPLANATION:
# Displays column names,
# data types,
# non-null values,
# memory usage.

print("\nDataset Information")
df.info()


# ============================================================
# SECTION 6 : STATISTICAL SUMMARY
# ============================================================

# CODE EXPLANATION:
# Generates descriptive statistics
# for all numerical columns.

print("\nStatistical Summary")
print(df.describe())


# ============================================================
# SECTION 7 : CHECK MISSING VALUES
# ============================================================

# CODE EXPLANATION:
# Counts missing values
# in every column.

print("\nMissing Values")
print(df.isnull().sum())


# ============================================================
# SECTION 8 : CHECK DUPLICATE VALUES
# ============================================================

# CODE EXPLANATION:
# Counts duplicate rows.

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)


# ============================================================
# SECTION 9 : DROP SERIAL NUMBER
# ============================================================

# CODE EXPLANATION:
# Serial Number is only an identifier.
# It does not contribute to prediction.

df.drop("Serial No.", axis=1, inplace=True)

print("\nUpdated Columns")
print(df.columns)


# ============================================================
# SECTION 10 : TARGET DISTRIBUTION
# ============================================================

# CODE EXPLANATION:
# Visualize the target variable.

plt.figure(figsize=(8,5))

sns.histplot(
    df["Chance of Admit "],
    bins=20,
    kde=True,
    color="royalblue"
)

plt.title("Chance of Admit Distribution")
plt.xlabel("Chance of Admit")
plt.ylabel("Frequency")

plt.show()


# ============================================================
# SECTION 11 : FEATURE DISTRIBUTION
# ============================================================

# CODE EXPLANATION:
# Display histogram
# for every numerical feature.

df.hist(
    figsize=(15,10),
    bins=20
)

plt.tight_layout()

plt.show()


# ============================================================
# SECTION 12 : CORRELATION HEATMAP
# ============================================================

# CODE EXPLANATION:
# Shows relationship
# between every feature.

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# ============================================================
# SECTION 13 : BOXPLOTS
# ============================================================

# CODE EXPLANATION:
# Detect potential outliers.

for column in df.columns:

    plt.figure(figsize=(7,3))

    sns.boxplot(
        x=df[column],
        color="lightgreen"
    )

    plt.title(column)

    plt.show()


# ============================================================
# SECTION 14 : PAIRPLOT (CPU FRIENDLY)
# ============================================================

# CODE EXPLANATION:
# Pairplot is computationally expensive.
# Sample only 150 records
# to improve performance.

sample_df = df.sample(
    n=150,
    random_state=42
)

sns.pairplot(sample_df)

plt.show()


# ============================================================
# SECTION 15 : EDA SUMMARY
# ============================================================

# CODE EXPLANATION:
# Display important observations
# from the exploratory analysis.

print("\n========== EDA SUMMARY ==========")

print("Total Samples :", len(df))
print("Total Features :", len(df.columns)-1)
print("Target Column : Chance of Admit")

print("\nNumerical Columns")
print(df.select_dtypes(include=np.number).columns.tolist())

print("\nDataset is ready for preprocessing.")
# ============================================================
#  DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================


# ============================================================
# SECTION 1 : FEATURE SELECTION
# ============================================================

# CODE EXPLANATION:
# The dataset contains one unnecessary column:
# Serial No.
#
# Since it is only an identifier,
# it does not contribute to prediction.
#
# Remove it before model training.

df.drop("Serial No.", axis=1, inplace=True)

print("Updated Columns")
print(df.columns)


# ============================================================
# SECTION 2 : CHECK MISSING VALUES
# ============================================================

# CODE EXPLANATION:
# Check whether any missing values
# exist after preprocessing.

print("\nMissing Values")

print(df.isnull().sum())


# ============================================================
# SECTION 3 : HANDLE MISSING VALUES
# ============================================================

# CODE EXPLANATION:
# Fill missing numerical values
# using median.
#
# Even if no missing values exist,
# this is a good industry practice.

df.fillna(
    df.median(numeric_only=True),
    inplace=True
)

print("\nMissing Values After Cleaning")

print(df.isnull().sum())


# ============================================================
# SECTION 4 : FEATURE & TARGET SPLIT
# ============================================================

# CODE EXPLANATION:
# Separate independent variables (X)
# and dependent variable (y).

X = df.drop(
    "Chance of Admit ",
    axis=1
)

y = df["Chance of Admit "]


print("\nFeature Shape :", X.shape)

print("Target Shape :", y.shape)


# ============================================================
# SECTION 5 : FEATURE NAMES
# ============================================================

# CODE EXPLANATION:
# Display all input features.

print("\nInput Features")

for feature in X.columns:

    print(feature)


# ============================================================
# SECTION 6 : TRAIN TEST SPLIT
# ============================================================

# CODE EXPLANATION:
# Split dataset into training
# and testing data.
#
# 80% → Training
# 20% → Testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples :", len(X_test))


# ============================================================
# SECTION 7 : FEATURE SCALING
# ============================================================

# CODE EXPLANATION:
# StandardScaler standardizes
# every feature.
#
# Mean = 0
#
# Standard Deviation = 1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


print("\nFeature Scaling Completed")


# ============================================================
# SECTION 8 : SAVE SCALER
# ============================================================

# CODE EXPLANATION:
# Save the fitted scaler.
#
# We will reuse it
# during deployment.

import joblib

joblib.dump(

    scaler,

    "scaler.pkl"

)

print("Scaler Saved Successfully")


# ============================================================
# SECTION 9 : CONVERT NUMPY TO PYTORCH TENSORS
# ============================================================

# CODE EXPLANATION:
# PyTorch models work with tensors,
# not NumPy arrays.

import torch

X_train_tensor = torch.FloatTensor(

    X_train

)

X_test_tensor = torch.FloatTensor(

    X_test

)

y_train_tensor = torch.FloatTensor(

    y_train.values

).reshape(-1,1)

y_test_tensor = torch.FloatTensor(

    y_test.values

).reshape(-1,1)


print("\nTensor Shapes")

print(X_train_tensor.shape)

print(y_train_tensor.shape)


# ============================================================
# SECTION 10 : UNDERSTANDING PYTORCH TENSORS
# ============================================================

# CODE EXPLANATION:
# Create a simple tensor
# to understand PyTorch.

tensor = torch.tensor(

    [1,2,3,4,5]

)

print("\nTensor")

print(tensor)

print("Shape :", tensor.shape)

print("Datatype :", tensor.dtype)

print("Device :", tensor.device)


# ============================================================
# SECTION 11 : BASIC TENSOR OPERATIONS
# ============================================================

# CODE EXPLANATION:
# Perform arithmetic operations.

a = torch.tensor(

    [10,20,30]

)

b = torch.tensor(

    [1,2,3]

)

print("\nAddition")

print(a+b)

print("\nSubtraction")

print(a-b)

print("\nMultiplication")

print(a*b)

print("\nDivision")

print(a/b)


# ============================================================
# SECTION 12 : MATRIX MULTIPLICATION
# ============================================================

# CODE EXPLANATION:
# Matrix multiplication
# using torch.matmul()

matrix1 = torch.tensor(

    [

        [1,2],

        [3,4]

    ]

)

matrix2 = torch.tensor(

    [

        [5,6],

        [7,8]

    ]

)

print("\nMatrix Multiplication")

print(

    torch.matmul(

        matrix1,

        matrix2

    )

)


# ============================================================
# SECTION 13 : TENSOR RESHAPING
# ============================================================

# CODE EXPLANATION:
# Change tensor dimensions.

tensor = torch.arange(12)

print("\nOriginal Tensor")

print(tensor)

print("\nReshaped Tensor")

print(

    tensor.reshape(

        3,

        4

    )

)


# ============================================================
# SECTION 14 : AUTOMATIC DIFFERENTIATION
# ============================================================

# CODE EXPLANATION:
# Autograd automatically computes
# gradients.

x = torch.tensor(

    5.0,

    requires_grad=True

)

y = x**2 + 2*x + 5

y.backward()

print("\nGradient")

print(x.grad)


# ============================================================
# SECTION 15 : PYTORCH DATASET
# ============================================================

# CODE EXPLANATION:
# TensorDataset combines
# features and labels.

from torch.utils.data import TensorDataset

train_dataset = TensorDataset(

    X_train_tensor,

    y_train_tensor

)

test_dataset = TensorDataset(

    X_test_tensor,

    y_test_tensor

)

print("\nTraining Dataset Size")

print(len(train_dataset))


# ============================================================
# SECTION 16 : DATALOADER
# ============================================================

# CODE EXPLANATION:
# DataLoader loads
# mini-batches.

from torch.utils.data import DataLoader

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True

)

test_loader = DataLoader(

    test_dataset,

    batch_size=32,

    shuffle=False

)

print("\nTraining Batches")

print(len(train_loader))


# ============================================================
# SECTION 17 : VERIFY FIRST BATCH
# ============================================================

# CODE EXPLANATION:
# Inspect one batch.

for features, labels in train_loader:

    print("\nFeature Batch Shape")

    print(features.shape)

    print("\nLabel Batch Shape")

    print(labels.shape)

    break


# ============================================================
# PART-2 SUMMARY
# ============================================================

print("===================================")

print("Data Preprocessed Successfully")

print("Features Scaled")

print("Scaler Saved")

print("Converted to PyTorch Tensors")

print("TensorDataset Created")

print("DataLoader Ready")

print("Everything is ready for Model Development.")

# ============================================================
#  MODEL DEVELOPMENT & EVALUATION
# PROJECT : Research Analysis using PyTorch
# ============================================================


# ============================================================
# SECTION 1 : IMPORT PYTORCH MODULES
# ============================================================

# CODE EXPLANATION:
# Import neural network modules,
# optimizer, and evaluation metrics.

import torch.nn as nn

import torch.optim as optim

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

import matplotlib.pyplot as plt


# ============================================================
# SECTION 2 : CHECK DEVICE
# ============================================================

# CODE EXPLANATION:
# Use GPU if available,
# otherwise CPU.

device = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)

print("Device :", device)


# ============================================================
# SECTION 3 : CREATE NEURAL NETWORK
# ============================================================

# CODE EXPLANATION:
# Build a Feed Forward Neural Network.

class AdmissionPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(7,64),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(64,32),

            nn.ReLU(),

            nn.Linear(32,16),

            nn.ReLU(),

            nn.Linear(16,1)

        )

    def forward(self,x):

        return self.network(x)


model = AdmissionPredictor().to(device)

print(model)


# ============================================================
# SECTION 4 : LOSS FUNCTION
# ============================================================

# CODE EXPLANATION:
# Mean Squared Error is commonly
# used for regression problems.

criterion = nn.MSELoss()


# ============================================================
# SECTION 5 : OPTIMIZER
# ============================================================

# CODE EXPLANATION:
# Adam optimizer updates
# model weights.

optimizer = optim.Adam(

    model.parameters(),

    lr=0.001

)


# ============================================================
# SECTION 6 : TRAINING SETTINGS
# ============================================================

epochs = 100

train_losses = []

validation_losses = []


# ============================================================
# SECTION 7 : TRAIN MODEL
# ============================================================

for epoch in range(epochs):

    model.train()

    epoch_loss = 0

    for features,labels in train_loader:

        features = features.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(features)

        loss = criterion(predictions,labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(

        epoch_loss / len(train_loader)

    )


    model.eval()

    validation_loss = 0

    with torch.no_grad():

        for features,labels in test_loader:

            features = features.to(device)

            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs,labels)

            validation_loss += loss.item()

    validation_losses.append(

        validation_loss / len(test_loader)

    )


    if (epoch+1)%10==0:

        print(

            f"Epoch {epoch+1}/{epochs}"

            f"  Train Loss : {train_losses[-1]:.5f}"

            f"  Validation Loss : {validation_losses[-1]:.5f}"

        )


# ============================================================
# SECTION 8 : LOSS CURVE
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(

    train_losses,

    label="Training Loss"

)

plt.plot(

    validation_losses,

    label="Validation Loss"

)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# SECTION 9 : MODEL PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(

        X_test_tensor.to(device)

    )

predictions = predictions.cpu().numpy()


# ============================================================
# SECTION 10 : EVALUATION METRICS
# ============================================================

mae = mean_absolute_error(

    y_test,

    predictions

)

mse = mean_squared_error(

    y_test,

    predictions

)

rmse = np.sqrt(mse)

r2 = r2_score(

    y_test,

    predictions

)

print()

print("MAE :", mae)

print("MSE :", mse)

print("RMSE :", rmse)

print("R2 Score :", r2)


# ============================================================
# SECTION 11 : ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7,6))

plt.scatter(

    y_test,

    predictions,

    alpha=0.7

)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("Actual vs Predicted")

plt.grid(True)

plt.show()


# ============================================================
# SECTION 12 : SAMPLE PREDICTIONS
# ============================================================

results = np.column_stack(

    (

        y_test.values,

        predictions.flatten()

    )

)

print()

print("First 10 Predictions")

print()

for i in range(10):

    print(

        f"Actual : {results[i][0]:.3f}"

        f"    Predicted : {results[i][1]:.3f}"

    )


# ============================================================
# SECTION 13 : SAVE MODEL
# ============================================================

torch.save(

    model.state_dict(),

    "research_model.pth"

)

print()

print("Model Saved Successfully")


# ============================================================
# SECTION 14 : LOAD MODEL
# ============================================================

loaded_model = AdmissionPredictor()

loaded_model.load_state_dict(

    torch.load(

        "research_model.pth",

        map_location=device

    )

)

loaded_model.eval()

print("Saved Model Loaded Successfully")


# ============================================================
# SECTION 15 : FINAL SUMMARY
# ============================================================


print("===================================")

print("Neural Network Built")

print("Training Completed")

print("Model Evaluated")

print("Regression Metrics Calculated")

print("Model Saved Successfully")

print("Ready For Deployment")

# ============================================================
#  MODEL DEPLOYMENT USING FASTAPI
# ============================================================


# ============================================================
# SECTION 1 : SAVE LABEL INFORMATION
# ============================================================

# CODE EXPLANATION:
# Save feature names to ensure the deployment
# uses the same input order as the training data.

feature_names = list(X.columns)

print("Feature Order")

for feature in feature_names:

    print(feature)


# ============================================================
# SECTION 2 : CREATE APP.PY
# ============================================================

# CODE EXPLANATION:
# The following code should be copied into
# a new file named app.py

app_code = '''

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np


# -----------------------------
# Load Scaler
# -----------------------------

scaler = joblib.load("scaler.pkl")


# -----------------------------
# Neural Network
# -----------------------------

class AdmissionPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(7,64),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(64,32),

            nn.ReLU(),

            nn.Linear(32,16),

            nn.ReLU(),

            nn.Linear(16,1)

        )

    def forward(self,x):

        return self.network(x)


model = AdmissionPredictor()

model.load_state_dict(

    torch.load(

        "research_model.pth",

        map_location=torch.device("cpu")

    )

)

model.eval()


app = FastAPI(

    title="Graduate Admission Prediction API",

    version="1.0"

)


class Student(BaseModel):

    GRE_Score: float

    TOEFL_Score: float

    University_Rating: float

    SOP: float

    LOR: float

    CGPA: float

    Research: float


@app.get("/")

def home():

    return {

        "message":"Graduate Admission Prediction API"

    }


@app.post("/predict")

def predict(student: Student):

    values = np.array([[
        student.GRE_Score,
        student.TOEFL_Score,
        student.University_Rating,
        student.SOP,
        student.LOR,
        student.CGPA,
        student.Research
    ]])

    values = scaler.transform(values)

    values = torch.FloatTensor(values)

    with torch.no_grad():

        prediction = model(values)

    return {

        "Chance_of_Admission":

        round(float(prediction.item()),4)

    }

'''

print(app_code)


# ============================================================
# SECTION 3 : SAVE APP FILE
# ============================================================

# CODE EXPLANATION:
# Save the deployment code into app.py

with open(

    "app.py",

    "w"

) as file:

    file.write(app_code)

print("app.py Created Successfully")


# ============================================================
# SECTION 4 : REQUIREMENTS.TXT
# ============================================================

requirements = """

torch
fastapi
uvicorn
numpy
pandas
scikit-learn
joblib
pydantic

"""

with open(

    "requirements.txt",

    "w"

) as file:

    file.write(requirements)

print("requirements.txt Created")


# ============================================================
# SECTION 5 : PROJECT STRUCTURE
# ============================================================

print("""

Research_Analysis_Using_PyTorch/

│

├── Admission_Predict.csv

├── Research_Analysis.ipynb

├── research_model.pth

├── scaler.pkl

├── app.py

├── requirements.txt

└── README.md

""")


# ============================================================
# SECTION 6 : RUN FASTAPI
# ============================================================

print("""

Run the following command inside terminal

uvicorn app:app --reload

""")


# ============================================================
# SECTION 7 : OPEN SWAGGER UI
# ============================================================

print("""

Open Browser

http://127.0.0.1:8000/docs

""")


# ============================================================
# SECTION 8 : SAMPLE INPUT
# ============================================================

sample = {

    "GRE_Score":337,

    "TOEFL_Score":118,

    "University_Rating":4,

    "SOP":4.5,

    "LOR":4.5,

    "CGPA":9.65,

    "Research":1

}

print(sample)


# ============================================================
# SECTION 9 : EXPECTED OUTPUT
# ============================================================

print("""

{

"Chance_of_Admission":0.94

}

""")


# ============================================================
# SECTION 10 : FINAL SUMMARY
# ============================================================

print()

print("========================================")

print("PROJECT COMPLETED SUCCESSFULLY")

print("========================================")

print("✔ Data Understanding")

print("✔ Data Preprocessing")

print("✔ PyTorch Fundamentals")

print("✔ Neural Network")

print("✔ Model Evaluation")

print("✔ Model Saved (.pth)")

print("✔ Scaler Saved (.pkl)")

print("✔ FastAPI Deployment")

print("✔ Swagger API")

print("✔ Production Ready")

print()

print("Congratulations!")

print("You have successfully built an End-to-End")

print("Graduate Admission Prediction System")

print("using PyTorch.")
