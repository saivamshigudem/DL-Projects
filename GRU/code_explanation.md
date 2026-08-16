# ============================================================
# FORECASTING USING GRU
#  DATA LOADING & EXPLORATORY DATA ANALYSIS
# CPU FRIENDLY
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 2. LOAD DATASET
# ============================================================

dataset_path = "HistoricalQuotes.csv"

df = pd.read_csv(dataset_path)


print("\n")
print("=" * 70)
print("GRU FORECASTING PROJECT")
print("=" * 70)

print("Dataset Loaded Successfully")

print(
    "Dataset Shape :",
    df.shape
)


# ============================================================
# 3. DISPLAY FIRST 5 RECORDS
# ============================================================

print("\n")
print("=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)

display(
    df.head()
)


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


print("\n")
print("=" * 70)
print("COLUMN NAMES")
print("=" * 70)

print(
    df.columns.tolist()
)


# ============================================================
# 5. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Close/Last",
    "Volume",
    "Open",
    "High",
    "Low"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if len(missing_columns) > 0:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print("\n")
print("=" * 70)
print("REQUIRED COLUMNS VERIFIED")
print("=" * 70)

print("All required columns are available.")


# ============================================================
# 6. CONVERT DATE COLUMN
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# ============================================================
# 7. CLEAN PRICE COLUMNS
# ============================================================

price_columns = [
    "Close/Last",
    "Open",
    "High",
    "Low"
]


for column in price_columns:

    df[column] = (

        df[column]
        .astype(str)
        .str.replace(
            "$",
            "",
            regex=False
        )
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.strip()
        .astype(float)

    )


# ============================================================
# 8. CLEAN VOLUME
# ============================================================

df["Volume"] = (

    df["Volume"]
    .astype(str)
    .str.replace(
        ",",
        "",
        regex=False
    )
    .str.strip()
    .astype(float)

)


# ============================================================
# 9. DISPLAY CLEANED DATA
# ============================================================

print("\n")
print("=" * 70)
print("CLEANED DATA")
print("=" * 70)

display(
    df.head()
)


# ============================================================
# 10. DATA TYPES
# ============================================================

print("\n")
print("=" * 70)
print("DATA TYPES")
print("=" * 70)

print(
    df.dtypes
)


# ============================================================
# 11. DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

df.info()


# ============================================================
# 12. MISSING VALUES
# ============================================================

missing_values = df.isnull().sum()


print("\n")
print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

display(
    missing_values
)


# ============================================================
# 13. DUPLICATE RECORDS
# ============================================================

duplicate_count = df.duplicated().sum()


print("\n")
print("=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

print(
    "Duplicate Rows :",
    duplicate_count
)


# ============================================================
# 14. REMOVE INVALID DATE RECORDS
# ============================================================

df = df.dropna(
    subset=[
        "Date"
    ]
)


# ============================================================
# 15. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)


print("\n")
print("=" * 70)
print("CHRONOLOGICAL ORDER")
print("=" * 70)

print("Oldest Records:")

display(
    df.head(3)
)

print("Latest Records:")

display(
    df.tail(3)
)


# ============================================================
# 16. STATISTICAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

display(
    df.describe()
)


# ============================================================
# 17. DATE RANGE
# ============================================================

print("\n")
print("=" * 70)
print("DATE RANGE")
print("=" * 70)

print(
    "Start Date :",
    df["Date"].min()
)

print(
    "End Date   :",
    df["Date"].max()
)


# ============================================================
# 18. CLOSING PRICE STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("CLOSING PRICE STATISTICS")
print("=" * 70)

print(
    "Minimum Close :",
    round(
        df["Close/Last"].min(),
        2
    )
)

print(
    "Maximum Close :",
    round(
        df["Close/Last"].max(),
        2
    )
)

print(
    "Average Close :",
    round(
        df["Close/Last"].mean(),
        2
    )
)

print(
    "Median Close  :",
    round(
        df["Close/Last"].median(),
        2
    )
)


# ============================================================
# 19. CLOSING PRICE TREND
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["Date"],
    df["Close/Last"],
    linewidth=1
)

plt.title(
    "Apple Stock Closing Price"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Closing Price ($)"
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 20. OPEN VS CLOSE
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["Date"],
    df["Open"],
    label="Open",
    linewidth=1
)

plt.plot(
    df["Date"],
    df["Close/Last"],
    label="Close",
    linewidth=1
)

plt.title(
    "Apple Open vs Closing Price"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Price ($)"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 21. TRADING VOLUME
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    df["Date"],
    df["Volume"],
    linewidth=1
)

plt.title(
    "Apple Trading Volume"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Volume"
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 22. CLOSING PRICE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    df["Close/Last"],
    bins=40,
    kde=True
)

plt.title(
    "Closing Price Distribution"
)

plt.xlabel(
    "Closing Price ($)"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.show()


# ============================================================
# 23. CLOSING PRICE BOX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)

sns.boxplot(
    y=df["Close/Last"]
)

plt.title(
    "Closing Price Box Plot"
)

plt.ylabel(
    "Closing Price ($)"
)

plt.tight_layout()

plt.show()


# ============================================================
# 24. CORRELATION MATRIX
# ============================================================

numeric_columns = [
    "Close/Last",
    "Volume",
    "Open",
    "High",
    "Low"
]

correlation_matrix = df[
    numeric_columns
].corr()


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Apple Stock Feature Correlation"
)

plt.tight_layout()

plt.show()


# ============================================================
# 25. INVALID VALUE CHECK
# ============================================================

print("\n")
print("=" * 70)
print("INVALID VALUE CHECK")
print("=" * 70)

print(
    "Negative Close Values :",
    (
        df["Close/Last"] < 0
    ).sum()
)

print(
    "Negative Volume Values:",
    (
        df["Volume"] < 0
    ).sum()
)


# ============================================================
# 26. FINAL DATASET SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Parameter": [

        "Dataset",
        "Problem Type",
        "Total Records",
        "Total Columns",
        "Target Variable",
        "Model",
        "Sequence Length",
        "Training Approach"
    ],

    "Value": [

        "Apple Historical Quotes",
        "Time Series Forecasting",
        len(df),
        len(df.columns),
        "Close/Last",
        "GRU",
        30,
        "Chronological Split"
    ]

})


print("\n")
print("=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

display(
    summary
)


# ============================================================
# 27. FINAL DATASET CHECK
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DATASET CHECK")
print("=" * 70)

print(
    "Rows           :",
    len(df)
)

print(
    "Columns        :",
    len(df.columns)
)

print(
    "Missing Values :",
    df.isnull().sum().sum()
)

print(
    "Duplicate Rows :",
    df.duplicated().sum()
)

print(
    "Start Date     :",
    df["Date"].min()
)

print(
    "End Date       :",
    df["Date"].max()
)

print(
    "Minimum Close  :",
    round(
        df["Close/Last"].min(),
        2
    )
)

print(
    "Maximum Close  :",
    round(
        df["Close/Last"].max(),
        2
    )
)


# ============================================================
# 28. COMPLETED
# ============================================================


print(
    "Dataset      : Apple Historical Quotes"
)

print(
    "Target       : Close/Last"
)

print(
    "Model        : GRU"
)

print(
    "Sequence     : 30 Trading Days"
)

print(
    "CPU Friendly : YES"
)

print("=" * 70)

# ============================================================
# GRU FORECASTING - PART-1B
# PREPROCESSING & SEQUENCE CREATION
# ============================================================

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Settings
SEQUENCE_LENGTH = 30
TRAIN_RATIO = 0.80
BATCH_SIZE = 16

# Target data
prices = df["Close/Last"].values.astype(np.float32).reshape(-1, 1)

# Chronological split
split = int(len(prices) * TRAIN_RATIO)

train_prices = prices[:split]
test_prices = prices[split:]

# Scale using ONLY training data
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_prices).astype(np.float32)
test_scaled = scaler.transform(test_prices).astype(np.float32)


# Create sequences
def create_sequences(data, length):
    X, y = [], []

    for i in range(len(data) - length):
        X.append(data[i:i + length])
        y.append(data[i + length])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Training sequences
X_train, y_train = create_sequences(
    train_scaled,
    SEQUENCE_LENGTH
)

# Include last 30 training prices for first test sequence
test_input = np.concatenate(
    [train_scaled[-SEQUENCE_LENGTH:], test_scaled]
)

X_test, y_test = create_sequences(
    test_input,
    SEQUENCE_LENGTH
)


# TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
).batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
).batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


# Check
print("=" * 60)
print("PART-1B COMPLETED")
print("=" * 60)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)
print("Sequence Length:", SEQUENCE_LENGTH)
print("Batch Size     :", BATCH_SIZE)

print("\nReady for GRU training.")

# ============================================================
# GRU FORECASTING - PART-2
# MODEL BUILDING & TRAINING
# ============================================================

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Build lightweight GRU
model = Sequential([
    GRU(
        32,
        activation="tanh",
        input_shape=(SEQUENCE_LENGTH, 1)
    ),
    Dense(16, activation="relu"),
    Dense(1)
])


# Compile
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_gru_model.keras",
    monitor="val_loss",
    save_best_only=True
)


# Train
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,
    callbacks=[early_stop, checkpoint],
    verbose=1
)


# Evaluate
loss, mae = model.evaluate(
    test_dataset,
    verbose=0
)


print("=" * 60)
print("PART-2 COMPLETED")
print("=" * 60)

print("Test Loss :", round(float(loss), 4))
print("Test MAE  :", round(float(mae), 4))


# Training curve
plt.figure(figsize=(8, 4))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("GRU Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.legend()
plt.grid()
plt.show()

# ============================================================
# GRU FORECASTING - PART-3
# EVALUATION & FUTURE FORECASTING
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error


# Predict test data
pred_scaled = model.predict(X_test, verbose=0)

# Convert back to actual prices
predicted = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_test)


# Metrics
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)

print("=" * 60)
print("GRU MODEL PERFORMANCE")
print("=" * 60)

print("MAE  :", round(float(mae), 2))
print("MSE  :", round(float(mse), 2))
print("RMSE :", round(float(rmse), 2))


# Actual vs Predicted
results = pd.DataFrame({
    "Actual": actual.flatten(),
    "Predicted": predicted.flatten()
})

display(results.head(10))


# Plot
plt.figure(figsize=(10, 4))

plt.plot(
    actual[:200],
    label="Actual"
)

plt.plot(
    predicted[:200],
    label="Predicted"
)

plt.title("GRU - Actual vs Predicted Price")
plt.xlabel("Trading Days")
plt.ylabel("Price ($)")

plt.legend()
plt.grid()
plt.show()


# ============================================================
# NEXT-DAY PREDICTION
# ============================================================

last_sequence = X_test[-1:]

next_scaled = model.predict(
    last_sequence,
    verbose=0
)

next_price = scaler.inverse_transform(
    next_scaled
)[0][0]

print("\n")
print("=" * 60)
print("NEXT DAY FORECAST")
print("=" * 60)

print(
    "Predicted Closing Price : $",
    round(float(next_price), 2)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    "GRU_Predictions.csv",
    index=False
)

pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE"],
    "Value": [mae, mse, rmse]
}).to_csv(
    "GRU_Metrics.csv",
    index=False
)


print("\n")
print("=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print("Model   : best_gru_model.keras")
print("Results : GRU_Predictions.csv")
print("Metrics : GRU_Metrics.csv")
print("Next Day Prediction : $", round(float(next_price), 2))
