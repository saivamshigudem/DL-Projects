# ============================================================
# STOCK PREDICTION USING LSTM
# PART-1A : DATA LOADING & EXPLORATORY DATA ANALYSIS
# CPU OPTIMIZED
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
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

print("Dataset Shape :", df.shape)


# ============================================================
# 3. DISPLAY FIRST 5 RECORDS
# ============================================================

print("\n")
print("=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)

display(df.head())


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\n")
print("=" * 70)
print("COLUMN NAMES")
print("=" * 70)

print(df.columns.tolist())


# ============================================================
# 5. CONVERT DATE COLUMN
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

print("\n")
print("=" * 70)
print("DATE CONVERSION")
print("=" * 70)

print("Date Data Type :", df["Date"].dtype)


# ============================================================
# 6. CLEAN PRICE COLUMNS
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
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )


# ============================================================
# 7. CLEAN VOLUME COLUMN
# ============================================================

df["Volume"] = (
    df["Volume"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
    .astype(float)
)


# ============================================================
# 8. DISPLAY CLEANED DATA
# ============================================================

print("\n")
print("=" * 70)
print("CLEANED DATA")
print("=" * 70)

display(df.head())


# ============================================================
# 9. DATA TYPES
# ============================================================

print("\n")
print("=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 10. DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

df.info()


# ============================================================
# 11. MISSING VALUES
# ============================================================

missing_values = df.isnull().sum()

print("\n")
print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

display(missing_values)


# ============================================================
# 12. DUPLICATE RECORDS
# ============================================================

duplicate_count = df.duplicated().sum()

print("\n")
print("=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

print("Number of Duplicate Rows :", duplicate_count)


# ============================================================
# 13. STATISTICAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

display(df.describe())


# ============================================================
# 14. SORT DATA CHRONOLOGICALLY
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

print("Oldest Record:")
display(df.head(3))

print("Latest Record:")
display(df.tail(3))


# ============================================================
# 15. DATASET DATE RANGE
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
# 16. CLOSING PRICE STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("CLOSING PRICE STATISTICS")
print("=" * 70)

print(
    "Minimum Closing Price :",
    round(df["Close/Last"].min(), 2)
)

print(
    "Maximum Closing Price :",
    round(df["Close/Last"].max(), 2)
)

print(
    "Average Closing Price :",
    round(df["Close/Last"].mean(), 2)
)

print(
    "Median Closing Price  :",
    round(df["Close/Last"].median(), 2)
)


# ============================================================
# 17. CLOSING PRICE VISUALIZATION
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
# 18. OPEN VS CLOSE PRICE
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
# 19. TRADING VOLUME
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
# 20. CLOSING PRICE DISTRIBUTION
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
    "Apple Closing Price Distribution"
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
# 21. CLOSING PRICE BOX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)

sns.boxplot(
    y=df["Close/Last"]
)

plt.title(
    "Apple Closing Price Box Plot"
)

plt.ylabel(
    "Closing Price ($)"
)

plt.tight_layout()

plt.show()


# ============================================================
# 22. CORRELATION MATRIX
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
# 23. CHECK INVALID VALUES
# ============================================================

print("\n")
print("=" * 70)
print("INVALID VALUE CHECK")
print("=" * 70)

print(
    "Negative Closing Prices :",
    (df["Close/Last"] < 0).sum()
)

print(
    "Negative Volume         :",
    (df["Volume"] < 0).sum()
)


# ============================================================
# 24. FINAL DATASET SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Parameter": [

        "Dataset",
        "Problem Type",
        "Total Records",
        "Total Features",
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
        "LSTM",
        30,
        "Chronological Split"
    ]

})

print("\n")
print("=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

display(summary)


# ============================================================
# 25. FINAL DATASET CHECK
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DATASET CHECK")
print("=" * 70)

print(
    "Rows              :",
    len(df)
)

print(
    "Columns           :",
    len(df.columns)
)

print(
    "Missing Values    :",
    df.isnull().sum().sum()
)

print(
    "Duplicate Rows    :",
    df.duplicated().sum()
)

print(
    "Start Date        :",
    df["Date"].min()
)

print(
    "End Date          :",
    df["Date"].max()
)

print(
    "Minimum Close     :",
    round(df["Close/Last"].min(), 2)
)

print(
    "Maximum Close     :",
    round(df["Close/Last"].max(), 2)
)

print(
    "Average Close     :",
    round(df["Close/Last"].mean(), 2)
)


# ============================================================
# 26.  COMPLETED
# ============================================================



print("Dataset           : Apple Historical Quotes")
print("Target            : Close/Last")
print("Model             : LSTM")
print("Sequence Length   : 30")
print("CPU Optimization  : Enabled")

print("=" * 70)

# ============================================================
# STOCK PREDICTION USING LSTM
# DATA PREPROCESSING & SEQUENCE GENERATION
# CPU OPTIMIZED
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30

TRAIN_RATIO = 0.80

BATCH_SIZE = 16


print("\n")
print("=" * 70)
print("LSTM PREPROCESSING CONFIGURATION")
print("=" * 70)

print("Sequence Length :", SEQUENCE_LENGTH)
print("Train Ratio     :", TRAIN_RATIO)
print("Batch Size      :", BATCH_SIZE)
print("Data Type       : float32")
print("CPU Optimization: Enabled")


# ============================================================
# 3. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Close/Last"
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


# ============================================================
# 4. CREATE CLEAN WORKING DATAFRAME
# ============================================================

stock_df = df[
    [
        "Date",
        "Close/Last"
    ]
].copy()


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

stock_df = stock_df.dropna(
    subset=[
        "Date",
        "Close/Last"
    ]
)


# ============================================================
# 6. SORT CHRONOLOGICALLY
# ============================================================

stock_df = stock_df.sort_values(
    "Date"
).reset_index(
    drop=True
)


print("\n")
print("=" * 70)
print("CLEAN DATASET")
print("=" * 70)

print(
    "Total Records :",
    len(stock_df)
)

print(
    "Start Date    :",
    stock_df["Date"].min()
)

print(
    "End Date      :",
    stock_df["Date"].max()
)


# ============================================================
# 7. EXTRACT CLOSE PRICE
# ============================================================

close_prices = stock_df[
    "Close/Last"
].values.astype(
    np.float32
)

close_prices = close_prices.reshape(
    -1,
    1
)


# ============================================================
# 8. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(close_prices) * TRAIN_RATIO
)

train_prices = close_prices[
    :split_index
]

test_prices = close_prices[
    split_index:
]


print("\n")
print("=" * 70)
print("CHRONOLOGICAL TRAIN / TEST SPLIT")
print("=" * 70)

print(
    "Total Samples   :",
    len(close_prices)
)

print(
    "Training Samples:",
    len(train_prices)
)

print(
    "Testing Samples :",
    len(test_prices)
)

print(
    "Training Start  :",
    stock_df["Date"].iloc[0]
)

print(
    "Training End    :",
    stock_df["Date"].iloc[split_index - 1]
)

print(
    "Testing Start   :",
    stock_df["Date"].iloc[split_index]
)

print(
    "Testing End     :",
    stock_df["Date"].iloc[-1]
)


# ============================================================
# 9. CREATE SCALER
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, 1)
)


# ============================================================
# 10. FIT SCALER ONLY ON TRAINING DATA
# ============================================================

train_scaled = scaler.fit_transform(
    train_prices
).astype(
    np.float32
)


# ============================================================
# 11. TRANSFORM TEST DATA
# ============================================================

test_scaled = scaler.transform(
    test_prices
).astype(
    np.float32
)


print("\n")
print("=" * 70)
print("NORMALIZATION COMPLETED")
print("=" * 70)

print(
    "Training Minimum :",
    round(
        float(train_scaled.min()),
        4
    )
)

print(
    "Training Maximum :",
    round(
        float(train_scaled.max()),
        4
    )
)

print(
    "Test Minimum     :",
    round(
        float(test_scaled.min()),
        4
    )
)

print(
    "Test Maximum     :",
    round(
        float(test_scaled.max()),
        4
    )
)


# ============================================================
# 12. CREATE SEQUENCES
# ============================================================

def create_sequences(
    data,
    sequence_length
):

    X = []
    y = []

    for i in range(
        len(data) - sequence_length
    ):

        X.append(
            data[
                i:i + sequence_length
            ]
        )

        y.append(
            data[
                i + sequence_length
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

    return X, y


# ============================================================
# 13. CREATE TRAINING SEQUENCES
# ============================================================

X_train, y_train = create_sequences(
    train_scaled,
    SEQUENCE_LENGTH
)


# ============================================================
# 14. CREATE TEST SEQUENCES
# ============================================================

# We need the last 30 training observations
# to provide historical context for the first test prediction.

test_input = np.concatenate(
    [
        train_scaled[
            -SEQUENCE_LENGTH:
        ],
        test_scaled
    ],
    axis=0
)


X_test, y_test = create_sequences(
    test_input,
    SEQUENCE_LENGTH
)


# ============================================================
# 15. VERIFY SHAPES
# ============================================================

print("\n")
print("=" * 70)
print("SEQUENCE GENERATION")
print("=" * 70)

print(
    "X_train Shape :",
    X_train.shape
)

print(
    "y_train Shape :",
    y_train.shape
)

print(
    "X_test Shape  :",
    X_test.shape
)

print(
    "y_test Shape  :",
    y_test.shape
)


# ============================================================
# 16. CHECK EXPECTED RNN/LSTM INPUT SHAPE
# ============================================================

print("\n")
print("=" * 70)
print("LSTM INPUT SHAPE")
print("=" * 70)

print(
    "Samples  :",
    X_train.shape[0]
)

print(
    "Time Steps:",
    X_train.shape[1]
)

print(
    "Features  :",
    X_train.shape[2]
)


# ============================================================
# 17. DISPLAY FIRST TRAINING SEQUENCE
# ============================================================

print("\n")
print("=" * 70)
print("FIRST TRAINING SEQUENCE")
print("=" * 70)

print(
    X_train[0].flatten()
)

print("\n")
print(
    "Target Value:",
    y_train[0][0]
)


# ============================================================
# 18. CREATE TENSORFLOW TRAIN DATASET
# ============================================================

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


# ============================================================
# 19. CREATE TENSORFLOW TEST DATASET
# ============================================================

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


# ============================================================
# 20. VERIFY BATCH
# ============================================================

sample_batch_X, sample_batch_y = next(
    iter(train_dataset)
)


print("\n")
print("=" * 70)
print("TENSORFLOW DATASET VERIFICATION")
print("=" * 70)

print(
    "Batch X Shape :",
    sample_batch_X.shape
)

print(
    "Batch y Shape :",
    sample_batch_y.shape
)


# ============================================================
# 21. DISPLAY SAMPLE ORIGINAL PRICES
# ============================================================

sample_scaled = X_train[0].reshape(
    -1,
    1
)

sample_original = scaler.inverse_transform(
    sample_scaled
)


sample_target = scaler.inverse_transform(
    y_train[0].reshape(
        -1,
        1
    )
)


print("\n")
print("=" * 70)
print("SAMPLE 30-DAY PRICE SEQUENCE")
print("=" * 70)

print(
    "Previous 30 Closing Prices:"
)

print(
    np.round(
        sample_original.flatten(),
        2
    )
)

print("\n")

print(
    "Actual Next Closing Price:",
    round(
        float(sample_target[0][0]),
        2
    )
)


# ============================================================
# 22. DATASET SUMMARY
# ============================================================

preprocessing_summary = pd.DataFrame({

    "Parameter": [

        "Dataset",
        "Target Variable",
        "Total Records",
        "Training Records",
        "Testing Records",
        "Sequence Length",
        "Features",
        "Batch Size",
        "Scaling",
        "Split Method",
        "Data Type"
    ],

    "Value": [

        "Apple Historical Quotes",
        "Close/Last",
        len(close_prices),
        len(train_prices),
        len(test_prices),
        SEQUENCE_LENGTH,
        1,
        BATCH_SIZE,
        "MinMaxScaler",
        "Chronological 80/20",
        "float32"
    ]

})


print("\n")
print("=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

display(
    preprocessing_summary
)


# ============================================================
# 23. FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL PREPROCESSING VALIDATION")
print("=" * 70)

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_test :",
    X_test.shape
)

print(
    "y_test :",
    y_test.shape
)

print(
    "Train Dataset Batches:",
    tf.data.experimental.cardinality(
        train_dataset
    ).numpy()
)

print(
    "Test Dataset Batches :",
    tf.data.experimental.cardinality(
        test_dataset
    ).numpy()
)

# ============================================================
# LSTM MODEL BUILDING, TRAINING & EVALUATION
# CPU OPTIMIZED
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# ============================================================
# 2. CPU CONFIGURATION
# ============================================================

# Keep TensorFlow CPU usage reasonable.
# You can change this to 2 if your laptop becomes slow.

try:

    tf.config.threading.set_intra_op_parallelism_threads(2)

    tf.config.threading.set_inter_op_parallelism_threads(2)

except RuntimeError:

    pass


print("\n")
print("=" * 70)
print("CPU CONFIGURATION")
print("=" * 70)

print("LSTM Units       : 32")
print("Dense Units      : 16")
print("Batch Size       :", BATCH_SIZE)
print("Sequence Length  :", SEQUENCE_LENGTH)
print("Epochs           : 10")
print("Optimizer        : Adam")
print("Device           : CPU")


# ============================================================
# 3. BUILD LIGHTWEIGHT LSTM MODEL
# ============================================================

model = Sequential([

    LSTM(
        32,
        activation="tanh",
        input_shape=(
            SEQUENCE_LENGTH,
            1
        )
    ),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1
    )

])


# ============================================================
# 4. DISPLAY MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LSTM MODEL")
print("=" * 70)

model.summary()


# ============================================================
# 5. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]

)


print("\n")
print("=" * 70)
print("MODEL COMPILED SUCCESSFULLY")
print("=" * 70)


# ============================================================
# 6. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True,

    verbose=1

)


model_checkpoint = ModelCheckpoint(

    "best_lstm_model.keras",

    monitor="val_loss",

    save_best_only=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001,

    verbose=1

)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\n")
print("=" * 70)
print("STARTING LSTM TRAINING")
print("=" * 70)


history = model.fit(

    train_dataset,

    validation_data=test_dataset,

    epochs=10,

    callbacks=[

        early_stopping,

        model_checkpoint,

        reduce_lr

    ],

    verbose=1

)


print("\n")
print("=" * 70)
print("LSTM TRAINING COMPLETED")
print("=" * 70)


# ============================================================
# 8. EVALUATE MODEL
# ============================================================

test_loss, test_mae_scaled = model.evaluate(

    test_dataset,

    verbose=0

)


print("\n")
print("=" * 70)
print("SCALED TEST RESULTS")
print("=" * 70)

print(
    "Test Loss :",
    round(
        float(test_loss),
        6
    )
)

print(
    "Test MAE  :",
    round(
        float(test_mae_scaled),
        6
    )
)


# ============================================================
# 9. GENERATE PREDICTIONS
# ============================================================

predictions_scaled = model.predict(

    X_test,

    verbose=0

)


# ============================================================
# 10. CONVERT PREDICTIONS TO ORIGINAL PRICE
# ============================================================

predictions_actual = scaler.inverse_transform(

    predictions_scaled

)


actual_prices = scaler.inverse_transform(

    y_test.reshape(
        -1,
        1
    )

)


# ============================================================
# 11. CALCULATE MAE
# ============================================================

mae = mean_absolute_error(

    actual_prices,

    predictions_actual

)


# ============================================================
# 12. CALCULATE MSE
# ============================================================

mse = mean_squared_error(

    actual_prices,

    predictions_actual

)


# ============================================================
# 13. CALCULATE RMSE
# ============================================================

rmse = np.sqrt(
    mse
)


# ============================================================
# 14. DISPLAY PERFORMANCE
# ============================================================

print("\n")
print("=" * 70)
print("LSTM PERFORMANCE")
print("=" * 70)

print(
    "MAE  :",
    round(
        float(mae),
        4
    )
)

print(
    "MSE  :",
    round(
        float(mse),
        4
    )
)

print(
    "RMSE :",
    round(
        float(rmse),
        4
    )
)


# ============================================================
# 15. CREATE PREDICTION DATAFRAME
# ============================================================

prediction_df = pd.DataFrame({

    "Actual Price":
        actual_prices.flatten(),

    "Predicted Price":
        predictions_actual.flatten()

})


print("\n")
print("=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)

display(
    prediction_df.head(20)
)


# ============================================================
# 16. TRAINING LOSS CURVE
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(

    history.history["loss"],

    label="Training Loss"

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"

)

plt.title(
    "LSTM Training vs Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE Loss"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 17. MAE CURVE
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(

    history.history["mae"],

    label="Training MAE"

)

plt.plot(

    history.history["val_mae"],

    label="Validation MAE"

)

plt.title(
    "LSTM Training vs Validation MAE"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MAE"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 18. ACTUAL VS PREDICTED PRICE
# ============================================================

# Show only first 200 test observations
# to keep the graph readable.

display_count = min(
    200,
    len(actual_prices)
)


plt.figure(
    figsize=(15, 6)
)

plt.plot(

    actual_prices[
        :display_count
    ],

    label="Actual Price"

)

plt.plot(

    predictions_actual[
        :display_count
    ],

    label="Predicted Price"

)

plt.title(
    "Apple Stock - Actual vs Predicted Price"
)

plt.xlabel(
    "Test Trading Days"
)

plt.ylabel(
    "Price ($)"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 19. PREDICTION ERROR
# ============================================================

errors = (

    actual_prices.flatten()

    -

    predictions_actual.flatten()

)


plt.figure(
    figsize=(10, 5)
)

plt.hist(

    errors,

    bins=30

)

plt.title(
    "LSTM Prediction Error Distribution"
)

plt.xlabel(
    "Prediction Error ($)"
)

plt.ylabel(
    "Frequency"
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 20. EVALUATION SUMMARY
# ============================================================

evaluation_summary = pd.DataFrame({

    "Metric": [

        "MAE",

        "MSE",

        "RMSE"

    ],

    "Value": [

        float(mae),

        float(mse),

        float(rmse)

    ]

})


print("\n")
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

display(
    evaluation_summary
)


# ============================================================
# 21. MODEL SUMMARY
# ============================================================

model_summary = pd.DataFrame({

    "Parameter": [

        "Dataset",

        "Problem",

        "Target",

        "Model",

        "LSTM Units",

        "Dense Units",

        "Sequence Length",

        "Batch Size",

        "Epochs",

        "Optimizer",

        "Loss",

        "MAE",

        "RMSE"

    ],

    "Value": [

        "Apple Historical Quotes",

        "Stock Price Forecasting",

        "Close/Last",

        "Lightweight LSTM",

        32,

        16,

        SEQUENCE_LENGTH,

        BATCH_SIZE,

        10,

        "Adam",

        "MSE",

        round(
            float(mae),
            4
        ),

        round(
            float(rmse),
            4
        )

    ]

})


print("\n")
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

display(
    model_summary
)


# ============================================================
# 22. BUSINESS / ML INSIGHTS
# ============================================================

print("\n")
print("=" * 70)
print("ML INSIGHTS")
print("=" * 70)

print(
    "1. LSTM can learn temporal patterns from previous prices."
)

print(
    "2. MinMax scaling helps stabilize neural network training."
)

print(
    "3. Chronological splitting prevents future-data leakage."
)

print(
    "4. MAE represents the average absolute prediction error."
)

print(
    "5. RMSE gives greater penalty to large prediction errors."
)

print(
    "6. EarlyStopping helps reduce unnecessary CPU training."
)

print(
    "7. A single lightweight LSTM is suitable for CPU experimentation."
)


# ============================================================
# 23. IMPORTANT MODEL LIMITATION
# ============================================================

print("\n")
print("=" * 70)
print("IMPORTANT LIMITATION")
print("=" * 70)

print(
    "This model learns historical price patterns."
)

print(
    "It does NOT guarantee future stock-market returns."
)

print(
    "Stock prices are affected by news, market conditions,"
)

print(
    "economic factors, company fundamentals, and investor sentiment."
)


# ============================================================
# 24 COMPLETED
# ============================================================



print(
    "Best Model Saved As: best_lstm_model.keras"
)

print(
    "Ready for Part-3 Deployment and Future Forecasting."
)

print("=" * 70)
# ============================================================
#  DEPLOYMENT & FUTURE FORECASTING
# CPU OPTIMIZED
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import load_model

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error


# ============================================================
# 2. LOAD BEST SAVED MODEL
# ============================================================

MODEL_PATH = "best_lstm_model.keras"


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"{MODEL_PATH} not found. "
        "Please run Part-2 first."
    )


loaded_model = load_model(
    MODEL_PATH
)


print("\n")
print("=" * 70)
print("LSTM MODEL LOADED SUCCESSFULLY")
print("=" * 70)

print(
    "Model File :",
    MODEL_PATH
)


# ============================================================
# 3. MODEL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("LOADED MODEL SUMMARY")
print("=" * 70)

loaded_model.summary()


# ============================================================
# 4. PREDICT TEST DATA
# ============================================================

test_predictions_scaled = loaded_model.predict(

    X_test,

    verbose=0

)


# ============================================================
# 5. CONVERT PREDICTIONS TO ORIGINAL PRICE
# ============================================================

test_predictions_actual = scaler.inverse_transform(

    test_predictions_scaled

)


test_actual_prices = scaler.inverse_transform(

    y_test.reshape(
        -1,
        1
    )

)


# ============================================================
# 6. CALCULATE FINAL METRICS
# ============================================================

final_mae = mean_absolute_error(

    test_actual_prices,

    test_predictions_actual

)


final_mse = mean_squared_error(

    test_actual_prices,

    test_predictions_actual

)


final_rmse = np.sqrt(
    final_mse
)


print("\n")
print("=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)

print(
    "MAE  :",
    round(
        float(final_mae),
        4
    )
)

print(
    "MSE  :",
    round(
        float(final_mse),
        4
    )
)

print(
    "RMSE :",
    round(
        float(final_rmse),
        4
    )
)


# ============================================================
# 7. CREATE TEST PREDICTION DATAFRAME
# ============================================================

test_prediction_df = pd.DataFrame({

    "Actual Price":
        test_actual_prices.flatten(),

    "Predicted Price":
        test_predictions_actual.flatten(),

    "Error":
        (
            test_actual_prices.flatten()
            -
            test_predictions_actual.flatten()
        )

})


print("\n")
print("=" * 70)
print("TEST PREDICTIONS")
print("=" * 70)

display(
    test_prediction_df.head(20)
)


# ============================================================
# 8. SAVE TEST PREDICTIONS
# ============================================================

test_prediction_df.to_csv(

    "LSTM_Test_Predictions.csv",

    index=False

)


print("\n")
print("=" * 70)
print("TEST PREDICTIONS SAVED")
print("=" * 70)

print(
    "File : LSTM_Test_Predictions.csv"
)


# ============================================================
# 9. SAVE EVALUATION METRICS
# ============================================================

evaluation_df = pd.DataFrame({

    "Metric": [

        "MAE",

        "MSE",

        "RMSE"

    ],

    "Value": [

        float(final_mae),

        float(final_mse),

        float(final_rmse)

    ]

})


evaluation_df.to_csv(

    "LSTM_Evaluation_Metrics.csv",

    index=False

)


print("\n")
print("=" * 70)
print("EVALUATION METRICS SAVED")
print("=" * 70)

display(
    evaluation_df
)


# ============================================================
# 10. ACTUAL VS PREDICTED GRAPH
# ============================================================

display_count = min(

    200,

    len(test_actual_prices)

)


plt.figure(

    figsize=(15, 6)

)


plt.plot(

    test_actual_prices[
        :display_count
    ],

    label="Actual Price"

)


plt.plot(

    test_predictions_actual[
        :display_count
    ],

    label="Predicted Price"

)


plt.title(

    "Apple Stock - Actual vs Predicted"

)


plt.xlabel(

    "Trading Days"

)


plt.ylabel(

    "Price ($)"

)


plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 11. NEXT TRADING DAY PREDICTION
# ============================================================

# Use the final 30 known scaled prices.

last_sequence = close_prices[
    -SEQUENCE_LENGTH:
]


# Scale using the scaler fitted during Part-1B.

last_sequence_scaled = scaler.transform(

    last_sequence

).astype(

    np.float32

)


# Reshape to:

# (1, 30, 1)

last_sequence_scaled = last_sequence_scaled.reshape(

    1,

    SEQUENCE_LENGTH,

    1

)


next_prediction_scaled = loaded_model.predict(

    last_sequence_scaled,

    verbose=0

)


next_prediction = scaler.inverse_transform(

    next_prediction_scaled

)


next_price = float(

    next_prediction[0][0]

)


last_known_price = float(

    close_prices[-1][0]

)


print("\n")
print("=" * 70)
print("NEXT TRADING DAY FORECAST")
print("=" * 70)

print(

    "Last Known Close : $",

    round(
        last_known_price,
        2
    )

)

print(

    "Predicted Next Close : $",

    round(
        next_price,
        2
    )

)


# ============================================================
# 12. CALCULATE PREDICTED CHANGE
# ============================================================

predicted_change = (

    next_price

    -

    last_known_price

)


predicted_change_percentage = (

    predicted_change

    /

    last_known_price

) * 100


print("\n")

print(
    "Predicted Change : $",
    round(
        predicted_change,
        2
    )
)


print(
    "Predicted Change % :",
    round(
        predicted_change_percentage,
        2
    ),
    "%"
)


# ============================================================
# 13. MULTI-DAY FORECASTING
# ============================================================

FORECAST_DAYS = 5


future_predictions = []


current_sequence = last_sequence_scaled.copy()


for day in range(

    FORECAST_DAYS

):

    prediction_scaled = loaded_model.predict(

        current_sequence,

        verbose=0

    )


    prediction_value = float(

        prediction_scaled[0][0]

    )


    prediction_actual = scaler.inverse_transform(

        prediction_scaled

    )[0][0]


    future_predictions.append(

        float(
            prediction_actual
        )

    )


    # Remove oldest observation
    # and append new prediction.

    new_value = np.array(

        [[[prediction_value]]],

        dtype=np.float32

    )


    current_sequence = np.concatenate(

        [

            current_sequence[:, 1:, :],

            new_value

        ],

        axis=1

    )


# ============================================================
# 14. CREATE FUTURE FORECAST DATAFRAME
# ============================================================

last_date = stock_df["Date"].iloc[-1]


future_dates = pd.bdate_range(

    start=last_date + pd.Timedelta(days=1),

    periods=FORECAST_DAYS

)


future_forecast_df = pd.DataFrame({

    "Forecast Date":
        future_dates,

    "Predicted Close":
        future_predictions

})


print("\n")
print("=" * 70)
print("5-DAY FUTURE FORECAST")
print("=" * 70)

display(
    future_forecast_df
)


# ============================================================
# 15. SAVE FUTURE FORECAST
# ============================================================

future_forecast_df.to_csv(

    "LSTM_Future_Forecast.csv",

    index=False

)


print("\n")
print("=" * 70)
print("FUTURE FORECAST SAVED")
print("=" * 70)

print(
    "File : LSTM_Future_Forecast.csv"
)


# ============================================================
# 16. FUTURE FORECAST VISUALIZATION
# ============================================================

recent_count = min(

    60,

    len(stock_df)

)


recent_dates = stock_df[

    "Date"

].iloc[

    -recent_count:

]


recent_prices = stock_df[

    "Close/Last"

].iloc[

    -recent_count:

]


plt.figure(

    figsize=(15, 6)

)


plt.plot(

    recent_dates,

    recent_prices,

    label="Historical Close"

)


plt.plot(

    future_forecast_df[
        "Forecast Date"
    ],

    future_forecast_df[
        "Predicted Close"
    ],

    marker="o",

    label="Forecast"

)


plt.title(

    "Apple Stock - Future LSTM Forecast"

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
# 17. CREATE DEPLOYMENT SUMMARY
# ============================================================

deployment_summary = pd.DataFrame({

    "Component": [

        "Dataset",

        "Target",

        "Model",

        "Sequence Length",

        "LSTM Units",

        "Batch Size",

        "Epochs",

        "MAE",

        "MSE",

        "RMSE",

        "Next Day Prediction",

        "Forecast Horizon"

    ],

    "Value": [

        "Apple Historical Quotes",

        "Close/Last",

        "Lightweight LSTM",

        SEQUENCE_LENGTH,

        32,

        BATCH_SIZE,

        10,

        round(
            float(final_mae),
            4
        ),

        round(
            float(final_mse),
            4
        ),

        round(
            float(final_rmse),
            4
        ),

        round(
            next_price,
            2
        ),

        f"{FORECAST_DAYS} trading days"

    ]

})


print("\n")
print("=" * 70)
print("DEPLOYMENT SUMMARY")
print("=" * 70)

display(
    deployment_summary
)


# ============================================================
# 18. SAVE DEPLOYMENT SUMMARY
# ============================================================

deployment_summary.to_csv(

    "LSTM_Deployment_Summary.csv",

    index=False

)


# ============================================================
# 19. PROJECT OUTPUT FILES
# ============================================================

output_files = pd.DataFrame({

    "Output": [

        "Trained Model",

        "Test Predictions",

        "Evaluation Metrics",

        "Future Forecast",

        "Deployment Summary"

    ],

    "File": [

        "best_lstm_model.keras",

        "LSTM_Test_Predictions.csv",

        "LSTM_Evaluation_Metrics.csv",

        "LSTM_Future_Forecast.csv",

        "LSTM_Deployment_Summary.csv"

    ]

})


print("\n")
print("=" * 70)
print("PROJECT OUTPUT FILES")
print("=" * 70)

display(
    output_files
)


# ============================================================
# 20. BUSINESS / ML INSIGHTS
# ============================================================

print("\n")
print("=" * 70)
print("ML INSIGHTS")
print("=" * 70)

print(
    "1. LSTM uses previous trading-day prices to forecast future prices."
)

print(
    "2. The model uses a 30-day historical window."
)

print(
    "3. MinMaxScaler helps keep input values in a neural-network-friendly range."
)

print(
    "4. Chronological splitting prevents future information leakage."
)

print(
    "5. MAE measures average absolute prediction error."
)

print(
    "6. RMSE penalizes larger prediction errors more strongly."
)

print(
    "7. Recursive forecasting becomes increasingly uncertain for longer horizons."
)


# ============================================================
# 21. IMPORTANT LIMITATION
# ============================================================

print("\n")
print("=" * 70)
print("IMPORTANT LIMITATION")
print("=" * 70)

print(
    "This LSTM is an educational forecasting model."
)

print(
    "It does NOT guarantee future stock prices or investment returns."
)

print(
    "Stock prices are influenced by many external factors,"
)

print(
    "including news, earnings, interest rates, market sentiment,"
)

print(
    "economic conditions, and unexpected events."
)


# ============================================================
# 22. FINAL PROJECT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STOCK PREDICTION USING LSTM")
print("=" * 70)

print(
    "Dataset             : Apple Historical Quotes"
)

print(
    "Target              : Close/Last"
)

print(
    "Model               : Lightweight LSTM"
)

print(
    "Sequence Length     :",
    SEQUENCE_LENGTH
)

print(
    "LSTM Units          : 32"
)

print(
    "Dense Units         : 16"
)

print(
    "Batch Size          :",
    BATCH_SIZE
)

print(
    "Epochs              : 10"
)

print(
    "Optimizer           : Adam"
)

print(
    "Loss                : MSE"
)

print(
    "MAE                 :",
    round(
        float(final_mae),
        4
    )
)

print(
    "RMSE                :",
    round(
        float(final_rmse),
        4
    )
)

print(
    "Next-Day Forecast   : $",
    round(
        next_price,
        2
    )
)

print(
    "Forecast Horizon    :",
    FORECAST_DAYS,
    "trading days"
)

print("=" * 70)


# ============================================================
# 23. PROJECT COMPLETED
# ============================================================

print("\n")

print("=" * 70)

print(
    "STOCK PREDICTION USING LSTM"
)

print(
    "PROJECT COMPLETED SUCCESSFULLY!"
)

print("=" * 70)

