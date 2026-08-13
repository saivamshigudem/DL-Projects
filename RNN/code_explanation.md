# ============================================================
# Sequence Prediction using Simple RNN
# Data Loading & Exploratory Data Analysis
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Load Dataset
# ============================================================

dataset_path = "city_temperature.csv"

df = pd.read_csv(

    dataset_path

)

print("\n")

print("="*70)

print("Dataset Loaded Successfully")

print("="*70)

# ============================================================
# Display First Five Rows
# ============================================================

print("\n")

print("="*70)

print("First Five Records")

print("="*70)

display(

    df.head()

)

# ============================================================
# Dataset Shape
# ============================================================

print("\n")

print("="*70)

print("Dataset Shape")

print("="*70)

print(

    "Rows :", df.shape[0]

)

print(

    "Columns :", df.shape[1]

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
# Data Types
# ============================================================

print("\n")

print("="*70)

print("Data Types")

print("="*70)

display(

    pd.DataFrame(

        df.dtypes,

        columns=["Data Type"]

    )

)

# ============================================================
# Missing Values
# ============================================================

missing_values = df.isnull().sum()

print("\n")

print("="*70)

print("Missing Values")

print("="*70)

display(

    missing_values[

        missing_values > 0

    ]

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

    duplicates

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
# Available Cities
# ============================================================

print("\n")

print("="*70)

print("Number of Cities")

print("="*70)

print(

    df["City"].nunique()

)

print("\n")

print("Sample Cities")

print(

    df["City"].unique()[:20]

)

# ============================================================
# Select One City (CPU Optimized)
# ============================================================

city_name = df["City"].value_counts().index[0]

city_df = df[

    df["City"] == city_name

].copy()

print("\n")

print("="*70)

print("Selected City")

print("="*70)

print(

    city_name

)

print(

    "Records :", len(city_df)

)

# ============================================================
# Remove Missing Temperatures
# ============================================================

city_df = city_df.dropna(

    subset=["AvgTemperature"]

)

# ============================================================
# Temperature Statistics
# ============================================================

print("\n")

print("="*70)

print("Temperature Statistics")

print("="*70)

display(

    city_df["AvgTemperature"].describe()

)

# ============================================================
# Temperature Distribution
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.histplot(

    city_df["AvgTemperature"],

    bins=30,

    kde=True

)

plt.title(

    "Temperature Distribution"

)

plt.xlabel(

    "Average Temperature"

)

plt.ylabel(

    "Frequency"

)

plt.show()

# ============================================================
# Time Series Plot
# ============================================================

plt.figure(

    figsize=(15,5)

)

plt.plot(

    city_df["AvgTemperature"].values,

    linewidth=1

)

plt.title(

    f"Temperature Trend - {city_name}"

)

plt.xlabel(

    "Days"

)

plt.ylabel(

    "Temperature"

)

plt.grid()

plt.show()

# ============================================================
# Box Plot
# ============================================================

plt.figure(

    figsize=(6,5)

)

sns.boxplot(

    y=city_df["AvgTemperature"]

)

plt.title(

    "Temperature Box Plot"

)

plt.show()

# ============================================================
# Correlation
# ============================================================

numeric_df = city_df.select_dtypes(

    include=np.number

)

plt.figure(

    figsize=(8,6)

)

sns.heatmap(

    numeric_df.corr(),

    annot=True,

    cmap="coolwarm"

)

plt.title(

    "Correlation Matrix"

)

plt.show()

# ============================================================
# Dataset Summary
# ============================================================

summary = pd.DataFrame({

    "Parameter":[

        "Dataset",

        "Selected City",

        "Total Records",

        "Temperature Column",

        "Problem Type",

        "Model"

    ],

    "Value":[

        "City Temperature",

        city_name,

        len(city_df),

        "AvgTemperature",

        "Sequence Prediction",

        "Simple RNN"

    ]

})

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

display(

    summary

)
# ============================================================
#  Data Preprocessing & Sequence Generation
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

# ============================================================
# Create Date Column
# ============================================================

city_df["Date"] = pd.to_datetime(

    city_df[["Year","Month","Day"]],

    errors="coerce"

)

city_df = city_df.dropna(

    subset=["Date"]

)

city_df = city_df.sort_values(

    "Date"

).reset_index(

    drop=True

)

print("\n")

print("="*70)

print("Date Column Created Successfully")

print("="*70)

display(

    city_df.head()

)

# ============================================================
# Keep Required Columns
# ============================================================

temperature_df = city_df[[

    "Date",

    "AvgTemperature"

]].copy()

# ============================================================
# Remove Invalid Temperatures
# ============================================================

temperature_df = temperature_df[

    temperature_df["AvgTemperature"] > -50

]

temperature_df = temperature_df.reset_index(

    drop=True

)

print("\n")

print("="*70)

print("Dataset after Cleaning")

print("="*70)

print(

    "Total Records :",

    len(

        temperature_df

    )

)

# ============================================================
# Normalize Temperature
# ============================================================

scaler = MinMaxScaler()

temperature_df["ScaledTemperature"] = scaler.fit_transform(

    temperature_df[["AvgTemperature"]]

)

print("\n")

print("="*70)

print("Normalization Completed")

print("="*70)

display(

    temperature_df.head()

)

# ============================================================
# Sequence Length
# ============================================================

SEQUENCE_LENGTH = 20

print("\n")

print("="*70)

print("Sequence Length")

print("="*70)

print(

    SEQUENCE_LENGTH

)

# ============================================================
# Create Sequences
# ============================================================

X = []

y = []

values = temperature_df["ScaledTemperature"].values

for i in range(

    len(values) - SEQUENCE_LENGTH

):

    X.append(

        values[

            i:i+SEQUENCE_LENGTH

        ]

    )

    y.append(

        values[

            i+SEQUENCE_LENGTH

        ]

    )

X = np.array(

    X,

    dtype=np.float32

)

y = np.array(

    y,

    dtype=np.float32

)

print("\n")

print("="*70)

print("Sequences Created")

print("="*70)

print(

    "X Shape :",

    X.shape

)

print(

    "y Shape :",

    y.shape

)

# ============================================================
# Reshape for RNN
# ============================================================

X = X.reshape(

    (

        X.shape[0],

        X.shape[1],

        1

    )

)

print("\n")

print("="*70)

print("Reshaped for RNN")

print("="*70)

print(

    X.shape

)

# ============================================================
# Train Test Split
# ============================================================

X_train,

X_test,

y_train,

y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    shuffle=False

)

print("\n")

print("="*70)

print("Dataset Split")

print("="*70)

print(

    "Training Samples :",

    len(

        X_train

    )

)

print(

    "Testing Samples :",

    len(

        X_test

    )

)

# ============================================================
# TensorFlow Dataset
# ============================================================

BATCH_SIZE = 16

train_dataset = tf.data.Dataset.from_tensor_slices(

    (

        X_train,

        y_train

    )

)

train_dataset = train_dataset.batch(

    BATCH_SIZE

).prefetch(

    tf.data.AUTOTUNE

)

test_dataset = tf.data.Dataset.from_tensor_slices(

    (

        X_test,

        y_test

    )

)

test_dataset = test_dataset.batch(

    BATCH_SIZE

).prefetch(

    tf.data.AUTOTUNE

)

print("\n")

print("="*70)

print("TensorFlow Dataset Created")

print("="*70)

# ============================================================
# Display Sample Sequence
# ============================================================

print("\n")

print("="*70)

print("Sample Input Sequence")

print("="*70)

print(

    X_train[0].flatten()

)

print("\n")

print("Target Value")

print(

    y_train[0]

)

# ============================================================
# Dataset Information
# ============================================================

dataset_information = pd.DataFrame({

    "Parameter":[

        "Selected City",

        "Sequence Length",

        "Training Samples",

        "Testing Samples",

        "Batch Size",

        "Feature"

    ],

    "Value":[

        city_name,

        SEQUENCE_LENGTH,

        len(X_train),

        len(X_test),

        BATCH_SIZE,

        "AvgTemperature"

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
# Dataset Ready
# ============================================================

print("\n")

print("="*70)

print("Dataset Ready for Simple RNN")

print("="*70)

# ============================================================
#  Model Building & Training
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import SimpleRNN

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

# ============================================================
# Build Simple RNN Model
# ============================================================

model = Sequential()

model.add(

    SimpleRNN(

        units=32,

        activation="tanh",

        input_shape=(SEQUENCE_LENGTH,1)

    )

)

model.add(

    Dense(

        16,

        activation="relu"

    )

)

model.add(

    Dense(

        1

    )

)

print("\n")

print("="*70)

print("Simple RNN Model Built Successfully")

print("="*70)

model.summary()

# ============================================================
# Compile Model
# ============================================================

model.compile(

    optimizer="adam",

    loss="mse",

    metrics=["mae"]

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

    patience=3,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "best_rnn_model.keras",

    monitor="val_loss",

    save_best_only=True

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

    validation_data=test_dataset,

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

print("Model Training Completed")

print("="*70)

# ============================================================
# Evaluate Model
# ============================================================

loss, mae = model.evaluate(

    test_dataset,

    verbose=0

)

print("\n")

print("="*70)

print("Evaluation Results")

print("="*70)

print("Loss :", round(loss,4))

print("MAE  :", round(mae,4))

# ============================================================
# Predictions
# ============================================================

predictions = model.predict(

    X_test,

    verbose=0

)

# ============================================================
# Convert Back to Original Scale
# ============================================================

predictions_actual = scaler.inverse_transform(

    predictions

)

y_test_actual = scaler.inverse_transform(

    y_test.reshape(-1,1)

)

# ============================================================
# Calculate Metrics
# ============================================================

mae_value = mean_absolute_error(

    y_test_actual,

    predictions_actual

)

mse_value = mean_squared_error(

    y_test_actual,

    predictions_actual

)

rmse_value = np.sqrt(

    mse_value

)

print("\n")

print("="*70)

print("Performance Metrics")

print("="*70)

print("MAE  :", round(mae_value,3))

print("MSE  :", round(mse_value,3))

print("RMSE :", round(rmse_value,3))

# ============================================================
# Plot Training Loss
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

plt.title("Training Loss")

plt.show()

# ============================================================
# Plot MAE
# ============================================================

plt.figure(

    figsize=(8,5)

)

plt.plot(

    history.history["mae"],

    label="Training MAE"

)

plt.plot(

    history.history["val_mae"],

    label="Validation MAE"

)

plt.legend()

plt.grid()

plt.title("Training MAE")

plt.show()

# ============================================================
# Actual vs Predicted
# ============================================================

plt.figure(

    figsize=(15,6)

)

plt.plot(

    y_test_actual[:200],

    label="Actual"

)

plt.plot(

    predictions_actual[:200],

    label="Predicted"

)

plt.legend()

plt.grid()

plt.title("Actual vs Predicted Temperature")

plt.show()

# ============================================================
# Prediction Results
# ============================================================

prediction_df = pd.DataFrame({

    "Actual Temperature":

        y_test_actual.flatten(),

    "Predicted Temperature":

        predictions_actual.flatten()

})

print("\n")

print("="*70)

print("Prediction Results")

print("="*70)

display(

    prediction_df.head(20)

)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- RNN learns sequential dependencies in time-series data.")

print("- Previous temperatures help predict future temperatures.")

print("- Normalization improves convergence.")

print("- EarlyStopping prevents overfitting.")

print("- Simple RNN is suitable for learning fundamentals before LSTM.")

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)

print("Simple RNN Summary")

print("="*70)

print("Dataset             : City Temperature")

print("Model               : Simple RNN")

print("Sequence Length     :", SEQUENCE_LENGTH)

print("RNN Units           : 32")

print("Batch Size          : 16")

print("Epochs              : 10")

print("Optimizer           : Adam")

print("Loss Function       : Mean Squared Error")

print("MAE                 :", round(mae_value,3))

print("RMSE                :", round(rmse_value,3))

print("="*70)

print("\n")

print("="*70)

# ============================================================
# Deployment
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import load_model

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = load_model(

    "best_rnn_model.keras"

)

print("\n")

print("="*70)

print("Simple RNN Model Loaded Successfully")

print("="*70)

# ============================================================
# Predict Test Dataset
# ============================================================

predictions = loaded_model.predict(

    X_test,

    verbose=0

)

# ============================================================
# Convert Predictions Back
# ============================================================

predictions_actual = scaler.inverse_transform(

    predictions

)

actual_values = scaler.inverse_transform(

    y_test.reshape(-1,1)

)

# ============================================================
# Performance Metrics
# ============================================================

mae = mean_absolute_error(

    actual_values,

    predictions_actual

)

mse = mean_squared_error(

    actual_values,

    predictions_actual

)

rmse = np.sqrt(

    mse

)

print("\n")

print("="*70)

print("Evaluation Metrics")

print("="*70)

print("MAE  :", round(mae,3))

print("MSE  :", round(mse,3))

print("RMSE :", round(rmse,3))

# ============================================================
# Predict Next Temperature
# ============================================================

last_sequence = X_test[-1]

last_sequence = np.expand_dims(

    last_sequence,

    axis=0

)

next_prediction = loaded_model.predict(

    last_sequence,

    verbose=0

)

next_temperature = scaler.inverse_transform(

    next_prediction

)

print("\n")

print("="*70)

print("Next Temperature Prediction")

print("="*70)

print(

    round(

        float(

            next_temperature[0][0]

        ),

        2

    )

)

# ============================================================
# Export Prediction Results
# ============================================================

prediction_df = pd.DataFrame({

    "Actual Temperature":

        actual_values.flatten(),

    "Predicted Temperature":

        predictions_actual.flatten()

})

prediction_df.to_csv(

    "RNN_Predictions.csv",

    index=False

)

print("\n")

print("="*70)

print("Predictions Saved Successfully")

print("="*70)

display(

    prediction_df.head(

        20

    )

)

# ============================================================
# Save Evaluation Metrics
# ============================================================

metrics_df = pd.DataFrame({

    "Metric":[

        "MAE",

        "MSE",

        "RMSE"

    ],

    "Value":[

        mae,

        mse,

        rmse

    ]

})

metrics_df.to_csv(

    "Evaluation_Metrics.csv",

    index=False

)

print("\n")

print("="*70)

print("Evaluation Metrics Saved")

print("="*70)

display(

    metrics_df

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
# Actual vs Predicted Visualization
# ============================================================

plt.figure(

    figsize=(15,6)

)

plt.plot(

    actual_values[:300],

    label="Actual"

)

plt.plot(

    predictions_actual[:300],

    label="Predicted"

)

plt.legend()

plt.grid()

plt.title(

    "Actual vs Predicted Temperature"

)

plt.show()

# ============================================================
# Error Distribution
# ============================================================

errors = actual_values.flatten() - predictions_actual.flatten()

plt.figure(

    figsize=(8,5)

)

plt.hist(

    errors,

    bins=30

)

plt.title(

    "Prediction Error Distribution"

)

plt.xlabel(

    "Prediction Error"

)

plt.ylabel(

    "Frequency"

)

plt.show()

# ============================================================
# Deployment Summary
# ============================================================

deployment_summary = pd.DataFrame({

    "File":[

        "Simple RNN Model",

        "Prediction Results",

        "Evaluation Metrics",

        "Training History"

    ],

    "Saved As":[

        "best_rnn_model.keras",

        "RNN_Predictions.csv",

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

print("- Simple RNN captures short-term temporal patterns.")

print("- Previous temperature observations influence future predictions.")

print("- Normalization stabilizes training.")

print("- MAE provides average prediction error.")

print("- RMSE penalizes large forecasting errors.")

print("- Sequence models are widely used in weather forecasting, finance, and demand prediction.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Sequence Prediction using Simple RNN")

print("="*70)

print("Dataset                 : city_temperature.csv")

print("Problem Type            : Time Series Forecasting")

print("Model                   : Simple RNN")

print("Sequence Length         :", SEQUENCE_LENGTH)

print("RNN Units               : 32")

print("Batch Size              : 16")

print("Epochs                  : 10")

print("Optimizer               : Adam")

print("Loss Function           : Mean Squared Error")

print("MAE                     :", round(mae,3))

print("RMSE                    :", round(rmse,3))

print("Model Saved             : best_rnn_model.keras")

print("Prediction File         : RNN_Predictions.csv")

print("Metrics File            : Evaluation_Metrics.csv")

print("Training History File   : Training_History.csv")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Sequence Prediction using Simple RNN")

print("Completed Successfully!")

print("="*70)





