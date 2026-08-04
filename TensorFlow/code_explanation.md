# ============================================================
# PART-1: EXPLORATORY DATA ANALYSIS (EDA)
#        + DATA PREPROCESSING
# PROJECT: California Housing Price Prediction using TensorFlow
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
import joblib

# ============================================================
# SECTION 1: LOAD DATASET
# ============================================================
# Reads Kaggle California Housing dataset.

df=pd.read_csv("housing.csv")

# ============================================================
# SECTION 2: DISPLAY FIRST 5 ROWS
# ============================================================
print(df.head())

# ============================================================
# SECTION 3: DATASET SHAPE
# ============================================================
print("Shape:",df.shape)

# ============================================================
# SECTION 4: COLUMN NAMES
# ============================================================
print(df.columns.tolist())

# ============================================================
# SECTION 5: DATASET INFORMATION
# ============================================================
df.info()

# ============================================================
# SECTION 6: STATISTICAL SUMMARY
# ============================================================
print(df.describe(include="all"))

# ============================================================
# SECTION 7: CHECK MISSING VALUES
# ============================================================
print(df.isnull().sum())

# ============================================================
# SECTION 8: HANDLE MISSING VALUES
# ============================================================
# Fill missing values in total_bedrooms using median.
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

# ============================================================
# SECTION 9: DUPLICATE VALUES
# ============================================================
print("Duplicate Rows:",df.duplicated().sum())

# ============================================================
# SECTION 10: TARGET DISTRIBUTION
# ============================================================
plt.figure(figsize=(8,5))
sns.histplot(df["median_house_value"],bins=40,kde=True)
plt.title("Median House Value Distribution")
plt.show()

# ============================================================
# SECTION 11: OCEAN PROXIMITY DISTRIBUTION
# ============================================================
plt.figure(figsize=(8,5))
sns.countplot(data=df,x="ocean_proximity")
plt.xticks(rotation=30)
plt.show()

# ============================================================
# SECTION 12: HISTOGRAMS
# ============================================================
df.hist(figsize=(15,10))
plt.tight_layout()
plt.show()

# ============================================================
# SECTION 13: BOXPLOTS
# ============================================================
num_cols=df.select_dtypes(include=np.number).columns
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# ============================================================
# SECTION 14: LABEL ENCODING
# ============================================================
encoder=LabelEncoder()
df["ocean_proximity"]=encoder.fit_transform(df["ocean_proximity"])

# ============================================================
# SECTION 15: CORRELATION HEATMAP
# ============================================================
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.show()

# ============================================================
# SECTION 16: FEATURE/TARGET SPLIT
# ============================================================
X=df.drop("median_house_value",axis=1)
y=df["median_house_value"]

# ============================================================
# SECTION 17: TRAIN TEST SPLIT
# ============================================================
X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.2,random_state=42)

# ============================================================
# SECTION 18: FEATURE SCALING
# ============================================================
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

joblib.dump(scaler,"scaler.pkl")

# ============================================================
# SECTION 19: FINAL VERIFICATION
# ============================================================
print("Training:",X_train.shape)
print("Testing :",X_test.shape)
print("Target Train:",y_train.shape)
print("Target Test :",y_test.shape)

print("Columns Used:")
print(['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity'])



# ============================================================
# PART-2: TENSORFLOW FOUNDATIONS + MODEL DEVELOPMENT
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler



# ============================================================
# SECTION 2: TENSORFLOW CONSTANTS
# ============================================================
a=tf.constant([[1,2],[3,4]])
b=tf.constant([[5,6],[7,8]])
print(a)
print(b)

# SECTION 3: VARIABLES
v=tf.Variable([[10.,20.],[30.,40.]])
print(v)

# SECTION 4: BASIC OPERATIONS
print(tf.add(a,b))
print(tf.subtract(a,b))
print(tf.multiply(a,b))
print(tf.matmul(a,b))

# SECTION 5: RESHAPE
print(tf.reshape(a,(4,)))

# SECTION 6: GRADIENT TAPE
x=tf.Variable(5.0)
with tf.GradientTape() as tape:
    y_grad=x**2+3*x+2
grad=tape.gradient(y_grad,x)
print("Gradient:",grad.numpy())

# SECTION 7: BUILD MODEL
model=Sequential([
Dense(64,activation="relu",input_shape=(X_train.shape[1],)),
Dropout(0.2),
Dense(32,activation="relu"),
Dense(16,activation="relu"),
Dense(1)
])

# SECTION 8: COMPILE
model.compile(optimizer="adam",loss="mse",metrics=["mae"])

# SECTION 9: SUMMARY
model.summary()

# SECTION 10: EARLY STOPPING
early=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

# SECTION 11: TRAIN
history=model.fit(
X_train,y_train,
validation_split=0.2,
epochs=50,
batch_size=32,
callbacks=[early],
verbose=1)

# SECTION 12: LOSS PLOT
plt.plot(history.history["loss"],label="Train")
plt.plot(history.history["val_loss"],label="Validation")
plt.legend()
plt.show()

# SECTION 13: MAE PLOT
plt.plot(history.history["mae"],label="Train")
plt.plot(history.history["val_mae"],label="Validation")
plt.legend()
plt.show()

# SECTION 14: PREDICT
pred=model.predict(X_test)

# SECTION 15: METRICS
mae=mean_absolute_error(y_test,pred)
mse=mean_squared_error(y_test,pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,pred)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2:",r2)

# SECTION 16: ACTUAL VS PREDICTED
plt.figure(figsize=(6,6))
plt.scatter(y_test,pred,s=10)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# SECTION 17: SAVE MODEL
model.save("housing_price_tf.keras")
print("Model Saved")

# SECTION 18: LOAD MODEL TEST
loaded=tf.keras.models.load_model("housing_price_tf.keras")
print(loaded.predict(X_test[:2]))

# ============================================================
# PART-3: MODEL DEPLOYMENT USING FASTAPI
# ============================================================

# Install:
# pip install fastapi uvicorn tensorflow joblib scikit-learn pandas numpy

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np

# ============================================================
# SECTION 1: LOAD MODEL AND SCALER
# ============================================================
model=tf.keras.models.load_model("housing_price_tf.keras")
scaler=joblib.load("scaler.pkl")

# ============================================================
# SECTION 2: CREATE FASTAPI APP
# ============================================================
app=FastAPI(
    title="California Housing Price Prediction API",
    version="1.0",
    description="TensorFlow Regression API"
)

# ============================================================
# SECTION 3: INPUT SCHEMA
# ============================================================
class HouseInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: int

# ============================================================
# SECTION 4: HOME ROUTE
# ============================================================
@app.get("/")
def home():
    return {"message":"Housing Price Prediction API is running"}

# ============================================================
# SECTION 5: PREDICTION ROUTE
# ============================================================
@app.post("/predict")
def predict(data:HouseInput):
    features=np.array([[
        data.longitude,
        data.latitude,
        data.housing_median_age,
        data.total_rooms,
        data.total_bedrooms,
        data.population,
        data.households,
        data.median_income,
        data.ocean_proximity
    ]])
    features=scaler.transform(features)
    pred=model.predict(features,verbose=0)
    return {"predicted_house_value":round(float(pred[0][0]),2)}

# ============================================================
# SECTION 6: RUN SERVER
# ============================================================
# uvicorn app:app --reload

# ============================================================
# SECTION 7: TEST
# ============================================================
# Open:
# http://127.0.0.1:8000/docs
#
# Sample JSON:
# {
# "longitude":-122.23,
# "latitude":37.88,
# "housing_median_age":41,
# "total_rooms":880,
# "total_bedrooms":129,
# "population":322,
# "households":126,
# "median_income":8.3252,
# "ocean_proximity":3
# }

# ============================================================
# SECTION 8: requirements.txt
# ============================================================
# tensorflow
# fastapi
# uvicorn
# numpy
# pandas
# scikit-learn
# joblib

# ============================================================
# PROJECT STRUCTURE
# ============================================================
# housing.csv
# scaler.pkl
# housing_price_tf.keras
# app.py
# requirements.txt
# README.md

# ============================================================
# END
# ============================================================


