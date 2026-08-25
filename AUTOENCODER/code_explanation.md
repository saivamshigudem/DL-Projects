# ============================================================
# PART-1 : DATASET LOADING & EDA
# PROJECT : ANOMALY DETECTION USING AUTOENCODERS
# FRAMEWORK : PYTORCH
# DATASET : SKLEARN BREAST CANCER DATASET
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer


print("Libraries Imported Successfully")


# ============================================================
# SECTION 2 : LOAD BUILT-IN DATASET
# ============================================================

# Load the Breast Cancer dataset directly
# from scikit-learn.

data = load_breast_cancer()


print()

print("Dataset Loaded Successfully")


# ============================================================
# SECTION 3 : CREATE DATAFRAME
# ============================================================

# Convert the numerical data into a Pandas DataFrame.

df = pd.DataFrame(

    data.data,

    columns=data.feature_names

)


# Add the target column.

df["target"] = data.target


print()

print("DataFrame Created Successfully")


# ============================================================
# SECTION 4 : DISPLAY DATASET
# ============================================================

print()

print("=" * 70)

print("FIRST 5 ROWS OF THE DATASET")

print("=" * 70)


display(

    df.head()

)


# ============================================================
# SECTION 5 : DATASET INFORMATION
# ============================================================

print()

print("=" * 70)

print("DATASET INFORMATION")

print("=" * 70)


print()

print("Dataset Shape:")

print(df.shape)


print()

print("Number of Rows:")

print(df.shape[0])


print()

print("Number of Columns:")

print(df.shape[1])


print()

print("Number of Features:")

print(len(data.feature_names))


# ============================================================
# SECTION 6 : FEATURE NAMES
# ============================================================

print()

print("=" * 70)

print("FEATURE NAMES")

print("=" * 70)


for index, feature in enumerate(data.feature_names):

    print(

        f"{index + 1}. {feature}"

    )


# ============================================================
# SECTION 7 : TARGET INFORMATION
# ============================================================

print()

print("=" * 70)

print("TARGET INFORMATION")

print("=" * 70)


print()

print("Target Names:")

print(data.target_names)


print()

print("Target Mapping:")

print()

print("0 -> Malignant")

print("1 -> Benign")


# ============================================================
# SECTION 8 : CHECK DATA TYPES
# ============================================================

print()

print("=" * 70)

print("DATA TYPES")

print("=" * 70)


print()

print(

    df.dtypes

)


# ============================================================
# SECTION 9 : CHECK MISSING VALUES
# ============================================================

print()

print("=" * 70)

print("MISSING VALUE ANALYSIS")

print("=" * 70)


missing_values = df.isnull().sum()


print()

print(

    missing_values

)


total_missing_values = missing_values.sum()


print()

print(

    "Total Missing Values:",

    total_missing_values

)


# ============================================================
# SECTION 10 : STATISTICAL SUMMARY
# ============================================================

print()

print("=" * 70)

print("STATISTICAL SUMMARY")

print("=" * 70)


display(

    df.describe()

)


# ============================================================
# SECTION 11 : CLASS DISTRIBUTION
# ============================================================

print()

print("=" * 70)

print("CLASS DISTRIBUTION")

print("=" * 70)


class_distribution = (

    df["target"]

    .value_counts()

    .sort_index()

)


print()

print("Malignant (0):")

print(

    class_distribution[0]

)


print()

print("Benign (1):")

print(

    class_distribution[1]

)


# ============================================================
# SECTION 12 : CREATE CLASS DISTRIBUTION DATAFRAME
# ============================================================

class_distribution_df = pd.DataFrame(

    {

        "Class": [

            "Malignant",

            "Benign"

        ],

        "Target": [

            0,

            1

        ],

        "Count": [

            class_distribution[0],

            class_distribution[1]

        ]

    }

)


print()

display(

    class_distribution_df

)


# ============================================================
# SECTION 13 : VISUALIZE CLASS DISTRIBUTION
# ============================================================

plt.figure(

    figsize=(7, 5)

)


plt.bar(

    class_distribution_df["Class"],

    class_distribution_df["Count"]

)


plt.title(

    "Breast Cancer Dataset Class Distribution"

)


plt.xlabel(

    "Class"

)


plt.ylabel(

    "Number of Samples"

)


plt.grid(

    axis="y",

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 14 : SELECT NORMAL AND ANOMALY DATA
# ============================================================

# For this anomaly detection project:
#
# Benign (1)    -> Normal Data
# Malignant (0) -> Anomaly Data


normal_df = df[

    df["target"] == 1

].copy()


anomaly_df = df[

    df["target"] == 0

].copy()


# Remove the target column because the Autoencoder
# should learn only from the numerical features.

normal_features = normal_df.drop(

    columns=["target"]

)


anomaly_features = anomaly_df.drop(

    columns=["target"]

)


print()

print("=" * 70)

print("NORMAL AND ANOMALY DATA")

print("=" * 70)


print()

print(

    "Normal Samples (Benign):",

    len(normal_features)

)


print()

print(

    "Anomaly Samples (Malignant):",

    len(anomaly_features)

)


print()

print(

    "Number of Input Features:",

    normal_features.shape[1]

)


# ============================================================
# SECTION 15 : DISPLAY NORMAL DATA
# ============================================================

print()

print("=" * 70)

print("NORMAL DATA SAMPLE")

print("=" * 70)


display(

    normal_features.head()

)


# ============================================================
# SECTION 16 : DISPLAY ANOMALY DATA
# ============================================================

print()

print("=" * 70)

print("ANOMALY DATA SAMPLE")

print("=" * 70)


display(

    anomaly_features.head()

)


# ============================================================
# SECTION 17 : FEATURE COMPARISON
# ============================================================

# Compare a few features between
# normal and anomaly samples.

selected_features = [

    "mean radius",

    "mean texture",

    "mean perimeter",

    "mean area",

    "mean smoothness"

]


for feature in selected_features:


    plt.figure(

        figsize=(8, 4)

    )


    plt.hist(

        normal_features[feature],

        bins=20,

        alpha=0.6,

        label="Normal (Benign)"

    )


    plt.hist(

        anomaly_features[feature],

        bins=20,

        alpha=0.6,

        label="Anomaly (Malignant)"

    )


    plt.title(

        f"Feature Distribution: {feature}"

    )


    plt.xlabel(

        feature

    )


    plt.ylabel(

        "Frequency"

    )


    plt.legend()


    plt.show()


# ============================================================
# SECTION 18 : CHECK FEATURE CORRELATION
# ============================================================

# Instead of displaying a large 30x30 correlation plot,
# we calculate correlations with the target.
# This is faster and easier to understand.


feature_target_correlation = (

    df

    .corr(numeric_only=True)["target"]

    .sort_values()

)


print()

print("=" * 70)

print("FEATURE CORRELATION WITH TARGET")

print("=" * 70)


print()

print(

    feature_target_correlation

)


# ============================================================
# SECTION 19 : TOP CORRELATED FEATURES
# ============================================================

top_correlated_features = (

    feature_target_correlation

    .drop("target")

    .abs()

    .sort_values(

        ascending=False

    )

    .head(10)

)


print()

print("=" * 70)

print("TOP 10 FEATURES CORRELATED WITH TARGET")

print("=" * 70)


print()

print(

    top_correlated_features

)


# ============================================================
# SECTION 20 : FINAL DATASET SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-1 DATASET SUMMARY")

print("=" * 70)


print()

print("Dataset Name: Breast Cancer Wisconsin Dataset")

print(

    "Total Samples:",

    len(df)

)

print(

    "Normal Samples:",

    len(normal_features)

)

print(

    "Anomaly Samples:",

    len(anomaly_features)

)

print(

    "Number of Features:",

    normal_features.shape[1]

)

print(

    "Missing Values:",

    total_missing_values

)


# ============================================================
# SECTION 21 : PART-1 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-1 COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("Important Variables Created:")

print()

print("✔ df")

print("✔ normal_df")

print("✔ anomaly_df")

print("✔ normal_features")

print("✔ anomaly_features")

print("✔ selected_features")

print("✔ data")

print()

print("READY FOR PART-2")

print("DATA PREPROCESSING & AUTOENCODER INPUT PREPARATION")

# ============================================================
# PART-2 : DATA PREPROCESSING & AUTOENCODER INPUT PREPARATION
# PROJECT : ANOMALY DETECTION USING AUTOENCODERS
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np

import torch

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


print("Part-2 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : SET RANDOM SEED
# ============================================================

# Setting a random seed helps us get more consistent results.

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)


print()

print("Random Seed Set Successfully")


# ============================================================
# SECTION 3 : SPLIT NORMAL DATA
# ============================================================

# The Autoencoder will learn only from normal samples.
#
# We split normal data into:
#
# 80% -> Training
# 20% -> Validation
#
# The validation data also contains only normal samples.


X_normal_train, X_normal_val = train_test_split(

    normal_features,

    test_size=0.20,

    random_state=RANDOM_STATE,

    shuffle=True

)


print()

print("=" * 60)

print("NORMAL DATA SPLIT")

print("=" * 60)


print()

print(

    "Normal Training Samples:",

    X_normal_train.shape[0]

)


print(

    "Normal Validation Samples:",

    X_normal_val.shape[0]

)


print(

    "Number of Features:",

    X_normal_train.shape[1]

)


# ============================================================
# SECTION 4 : PREPARE ANOMALY DATA
# ============================================================

# Anomaly data is not used for training.
#
# It will be used later to check whether the Autoencoder
# can identify abnormal samples.


X_anomaly = anomaly_features.copy()


print()

print("=" * 60)

print("ANOMALY DATA")

print("=" * 60)


print()

print(

    "Anomaly Samples:",

    X_anomaly.shape[0]

)


print(

    "Number of Features:",

    X_anomaly.shape[1]

)


# ============================================================
# SECTION 5 : CREATE FEATURE SCALER
# ============================================================

# Autoencoders work better when numerical features
# are on a similar scale.
#
# StandardScaler transforms the data so that
# each feature has approximately:
#
# Mean = 0
# Standard Deviation = 1


scaler = StandardScaler()


# ============================================================
# SECTION 6 : FIT SCALER ON NORMAL TRAINING DATA
# ============================================================

# IMPORTANT:
#
# We fit the scaler ONLY on training data.
#
# This prevents data leakage.


X_normal_train_scaled = scaler.fit_transform(

    X_normal_train

)


# ============================================================
# SECTION 7 : SCALE VALIDATION DATA
# ============================================================

# We use transform() only.
#
# We do not fit the scaler again.


X_normal_val_scaled = scaler.transform(

    X_normal_val

)


# ============================================================
# SECTION 8 : SCALE ANOMALY DATA
# ============================================================

# Apply the same scaling learned from normal training data.


X_anomaly_scaled = scaler.transform(

    X_anomaly

)


print()

print("Feature Scaling Completed Successfully")


# ============================================================
# SECTION 9 : CHECK SCALED DATA
# ============================================================

print()

print("=" * 60)

print("SCALED DATA INFORMATION")

print("=" * 60)


print()

print(

    "Normal Training Shape:",

    X_normal_train_scaled.shape

)


print(

    "Normal Validation Shape:",

    X_normal_val_scaled.shape

)


print(

    "Anomaly Data Shape:",

    X_anomaly_scaled.shape

)


print()

print(

    "Training Data Mean:",

    round(

        X_normal_train_scaled.mean(),

        4

    )

)


print(

    "Training Data Standard Deviation:",

    round(

        X_normal_train_scaled.std(),

        4

    )

)


# ============================================================
# SECTION 10 : CONVERT NUMPY ARRAYS TO PYTORCH TENSORS
# ============================================================

# float32 is memory-efficient and standard for
# neural network training.


X_train_tensor = torch.tensor(

    X_normal_train_scaled,

    dtype=torch.float32

)


X_val_tensor = torch.tensor(

    X_normal_val_scaled,

    dtype=torch.float32

)


X_anomaly_tensor = torch.tensor(

    X_anomaly_scaled,

    dtype=torch.float32

)


print()

print("PyTorch Tensor Conversion Completed")


# ============================================================
# SECTION 11 : CHECK TENSOR SHAPES
# ============================================================

print()

print("=" * 60)

print("TENSOR INFORMATION")

print("=" * 60)


print()

print(

    "Training Tensor Shape:",

    X_train_tensor.shape

)


print(

    "Validation Tensor Shape:",

    X_val_tensor.shape

)


print(

    "Anomaly Tensor Shape:",

    X_anomaly_tensor.shape

)


print()

print(

    "Tensor Data Type:",

    X_train_tensor.dtype

)


# ============================================================
# SECTION 12 : CREATE PYTORCH DATASETS
# ============================================================

# Autoencoder input and target are the same.
#
# Input:
# X
#
# Target:
# X
#
# The model learns:
#
# Input -> Encoder -> Latent Space -> Decoder -> Reconstruction


train_dataset = TensorDataset(

    X_train_tensor,

    X_train_tensor

)


val_dataset = TensorDataset(

    X_val_tensor,

    X_val_tensor

)


# ============================================================
# SECTION 13 : CREATE DATALOADERS
# ============================================================

# Small batch size for slow CPU.


BATCH_SIZE = 16


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


print()

print("DataLoaders Created Successfully")


# ============================================================
# SECTION 14 : CHECK DATALOADER INFORMATION
# ============================================================

print()

print("=" * 60)

print("DATALOADER INFORMATION")

print("=" * 60)


print()

print(

    "Batch Size:",

    BATCH_SIZE

)


print(

    "Training Batches:",

    len(train_loader)

)


print(

    "Validation Batches:",

    len(val_loader)

)


# ============================================================
# SECTION 15 : INSPECT ONE BATCH
# ============================================================

# Get one batch from the training DataLoader.


sample_inputs, sample_targets = next(

    iter(train_loader)

)


print()

print("=" * 60)

print("SAMPLE BATCH INFORMATION")

print("=" * 60)


print()

print(

    "Input Batch Shape:",

    sample_inputs.shape

)


print(

    "Target Batch Shape:",

    sample_targets.shape

)


# The input and target should be identical
# for an Autoencoder.


print()

print(

    "Input and Target Are Equal:",

    torch.equal(

        sample_inputs,

        sample_targets

    )

)


# ============================================================
# SECTION 16 : CHECK DEVICE
# ============================================================

# Check whether GPU is available.
# Your system will most likely use CPU.


device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print()

print("=" * 60)

print("COMPUTATION DEVICE")

print("=" * 60)


print()

print(

    "Using Device:",

    device

)


# ============================================================
# SECTION 17 : DEFINE INPUT DIMENSION
# ============================================================

# The Autoencoder input size is equal to
# the number of features.


INPUT_DIM = X_train_tensor.shape[1]


print()

print(

    "Autoencoder Input Dimension:",

    INPUT_DIM

)


# ============================================================
# SECTION 18 : FINAL DATA PREPARATION SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-2 DATA PREPARATION SUMMARY")

print("=" * 70)


print()

print(

    "Normal Training Samples:",

    len(X_train_tensor)

)


print(

    "Normal Validation Samples:",

    len(X_val_tensor)

)


print(

    "Anomaly Samples:",

    len(X_anomaly_tensor)

)


print(

    "Input Features:",

    INPUT_DIM

)


print(

    "Batch Size:",

    BATCH_SIZE

)


print(

    "Device:",

    device

)


# ============================================================
# SECTION 19 : PART-2 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-2 COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("Important Variables Created:")

print()

print("✔ X_train_tensor")

print("✔ X_val_tensor")

print("✔ X_anomaly_tensor")

print("✔ train_dataset")

print("✔ val_dataset")

print("✔ train_loader")

print("✔ val_loader")

print("✔ scaler")

print("✔ INPUT_DIM")

print("✔ device")

print()

print("READY FOR PART-3")

print("AUTOENCODER DEVELOPMENT & TRAINING")
# ============================================================
# PART-3 : AUTOENCODER DEVELOPMENT & TRAINING
# PROJECT : ANOMALY DETECTION USING AUTOENCODERS
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import torch

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt


print("Part-3 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : SET AUTOENCODER PARAMETERS
# ============================================================

# Small architecture for CPU-friendly training.

HIDDEN_DIM = 16

LATENT_DIM = 8

LEARNING_RATE = 0.001

EPOCHS = 50


print()

print("=" * 60)

print("AUTOENCODER CONFIGURATION")

print("=" * 60)


print()

print("Input Dimension:", INPUT_DIM)

print("Hidden Dimension:", HIDDEN_DIM)

print("Latent Dimension:", LATENT_DIM)

print("Learning Rate:", LEARNING_RATE)

print("Epochs:", EPOCHS)


# ============================================================
# SECTION 3 : CREATE AUTOENCODER MODEL
# ============================================================

class Autoencoder(nn.Module):


    def __init__(

        self,

        input_dim,

        hidden_dim,

        latent_dim

    ):

        super().__init__()


        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------
        #
        # Compresses:
        #
        # 30 Features
        #      ↓
        # 16 Features
        #      ↓
        # 8 Latent Features
        #

        self.encoder = nn.Sequential(

            nn.Linear(

                input_dim,

                hidden_dim

            ),

            nn.ReLU(),

            nn.Linear(

                hidden_dim,

                latent_dim

            ),

            nn.ReLU()

        )


        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------
        #
        # Reconstructs:
        #
        # 8 Latent Features
        #      ↓
        # 16 Features
        #      ↓
        # 30 Reconstructed Features
        #

        self.decoder = nn.Sequential(

            nn.Linear(

                latent_dim,

                hidden_dim

            ),

            nn.ReLU(),

            nn.Linear(

                hidden_dim,

                input_dim

            )

        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(

        self,

        x

    ):


        # Encode input into compressed representation.

        encoded = self.encoder(

            x

        )


        # Decode compressed representation.

        decoded = self.decoder(

            encoded

        )


        return decoded


print()

print("Autoencoder Class Created Successfully")


# ============================================================
# SECTION 4 : CREATE MODEL
# ============================================================

model = Autoencoder(

    input_dim=INPUT_DIM,

    hidden_dim=HIDDEN_DIM,

    latent_dim=LATENT_DIM

)


# Move model to CPU or GPU.

model = model.to(

    device

)


print()

print("=" * 60)

print("AUTOENCODER MODEL")

print("=" * 60)


print()

print(model)


# ============================================================
# SECTION 5 : COUNT TRAINABLE PARAMETERS
# ============================================================

total_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad

)


print()

print("Total Trainable Parameters:")

print(total_parameters)


# ============================================================
# SECTION 6 : DEFINE LOSS FUNCTION
# ============================================================

# Mean Squared Error measures the difference between:
#
# Original Input
#
# and
#
# Reconstructed Output

criterion = nn.MSELoss()


print()

print("Loss Function: Mean Squared Error")


# ============================================================
# SECTION 7 : DEFINE OPTIMIZER
# ============================================================

optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


print()

print("Optimizer: Adam")


# ============================================================
# SECTION 8 : CREATE TRAINING HISTORY
# ============================================================

train_losses = []

val_losses = []


# ============================================================
# SECTION 9 : TRAIN THE AUTOENCODER
# ============================================================

print()

print("=" * 60)

print("STARTING AUTOENCODER TRAINING")

print("=" * 60)


for epoch in range(EPOCHS):


    # Put model in training mode.

    model.train()


    running_train_loss = 0.0


    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for inputs, targets in train_loader:


        # Move data to selected device.

        inputs = inputs.to(

            device

        )


        targets = targets.to(

            device

        )


        # Clear old gradients.

        optimizer.zero_grad()


        # Forward pass.

        reconstructed = model(

            inputs

        )


        # Calculate reconstruction loss.

        loss = criterion(

            reconstructed,

            targets

        )


        # Backpropagation.

        loss.backward()


        # Update model parameters.

        optimizer.step()


        # Add batch loss.

        running_train_loss += loss.item()


    # Calculate average training loss.

    epoch_train_loss = (

        running_train_loss /

        len(train_loader)

    )


    train_losses.append(

        epoch_train_loss

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    running_val_loss = 0.0


    # Disable gradient calculation.

    with torch.no_grad():


        for inputs, targets in val_loader:


            inputs = inputs.to(

                device

            )


            targets = targets.to(

                device

            )


            # Reconstruct validation samples.

            reconstructed = model(

                inputs

            )


            # Calculate validation loss.

            val_loss = criterion(

                reconstructed,

                targets

            )


            running_val_loss += val_loss.item()


    # Calculate average validation loss.

    epoch_val_loss = (

        running_val_loss /

        len(val_loader)

    )


    val_losses.append(

        epoch_val_loss

    )


    # Print progress every 5 epochs.

    if (

        (epoch + 1) % 5 == 0

        or

        epoch == 0

    ):


        print(

            f"Epoch [{epoch + 1}/{EPOCHS}] "

            f"| Train Loss: {epoch_train_loss:.6f} "

            f"| Validation Loss: {epoch_val_loss:.6f}"

        )


# ============================================================
# SECTION 10 : TRAINING COMPLETED
# ============================================================

print()

print("=" * 60)

print("AUTOENCODER TRAINING COMPLETED")

print("=" * 60)


print()

print(

    "Final Training Loss:",

    round(

        train_losses[-1],

        6

    )

)


print(

    "Final Validation Loss:",

    round(

        val_losses[-1],

        6

    )

)


# ============================================================
# SECTION 11 : VISUALIZE TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.plot(

    train_losses,

    label="Training Loss"

)


plt.plot(

    val_losses,

    label="Validation Loss"

)


plt.title(

    "Autoencoder Training and Validation Loss"

)


plt.xlabel(

    "Epoch"

)


plt.ylabel(

    "Mean Squared Error"

)


plt.legend()


plt.grid(

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 12 : TEST ONE RECONSTRUCTION
# ============================================================

model.eval()


# Select one normal validation sample.

sample_input = X_val_tensor[0].unsqueeze(

    0

).to(

    device

)


with torch.no_grad():


    sample_reconstruction = model(

        sample_input

    )


# Move tensors back to CPU.

original_sample = sample_input.cpu().numpy().flatten()

reconstructed_sample = (

    sample_reconstruction.cpu().numpy().flatten()

)


# Calculate reconstruction error.

sample_error = np.mean(

    (

        original_sample -

        reconstructed_sample

    ) ** 2

)


print()

print("=" * 60)

print("SAMPLE RECONSTRUCTION")

print("=" * 60)


print()

print(

    "Reconstruction Error:",

    round(

        sample_error,

        6

    )

)


# ============================================================
# SECTION 13 : COMPARE ORIGINAL AND RECONSTRUCTED FEATURES
# ============================================================

# To keep visualization simple,
# compare the first 10 features.

feature_count = 10


plt.figure(

    figsize=(12, 5)

)


plt.plot(

    range(feature_count),

    original_sample[:feature_count],

    marker="o",

    label="Original"

)


plt.plot(

    range(feature_count),

    reconstructed_sample[:feature_count],

    marker="x",

    label="Reconstructed"

)


plt.title(

    "Original vs Reconstructed Features"

)


plt.xlabel(

    "Feature Index"

)


plt.ylabel(

    "Scaled Feature Value"

)


plt.legend()


plt.grid(

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 14 : SAVE TRAINED MODEL
# ============================================================

MODEL_PATH = "autoencoder_anomaly_detection.pth"


torch.save(

    model.state_dict(),

    MODEL_PATH

)


print()

print("Model Saved Successfully")

print("Model Path:", MODEL_PATH)


# ============================================================
# SECTION 15 : PART-3 SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-3 COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("Completed:")

print()

print("✔ Autoencoder Architecture")

print("✔ Encoder")

print("✔ Latent Space")

print("✔ Decoder")

print("✔ MSE Reconstruction Loss")

print("✔ Model Training")

print("✔ Validation")

print("✔ Training Visualization")

print("✔ Sample Reconstruction")

print("✔ Model Saving")


print()

print("READY FOR PART-4")

print("ANOMALY DETECTION & MODEL EVALUATION")
# ============================================================
# PART-4 : ANOMALY DETECTION & MODEL EVALUATION
# PROJECT : ANOMALY DETECTION USING AUTOENCODERS
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np

import torch

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


print("Part-4 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : LOAD TRAINED MODEL WEIGHTS
# ============================================================

# Load the model weights saved in Part-3.

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


# Move model to the selected device.

model = model.to(device)


# Put model in evaluation mode.

model.eval()


print()

print("Trained Autoencoder Loaded Successfully")


# ============================================================
# SECTION 3 : FUNCTION TO CALCULATE RECONSTRUCTION ERROR
# ============================================================

def calculate_reconstruction_errors(data_tensor, model):


    # Put model in evaluation mode.

    model.eval()


    reconstruction_errors = []


    # Disable gradient calculation.

    with torch.no_grad():


        # Process one sample at a time.
        # This is completely fine because the dataset is small.

        for sample in data_tensor:


            # Add batch dimension.

            sample = sample.unsqueeze(0).to(device)


            # Reconstruct the input.

            reconstructed = model(sample)


            # Calculate Mean Squared Error.

            error = torch.mean(
                (
                    sample - reconstructed
                ) ** 2
            ).item()


            # Store the reconstruction error.

            reconstruction_errors.append(
                error
            )


    return np.array(
        reconstruction_errors
    )


print()

print("Reconstruction Error Function Created Successfully")


# ============================================================
# SECTION 4 : CALCULATE NORMAL VALIDATION ERRORS
# ============================================================

print()

print("Calculating Normal Validation Reconstruction Errors...")


normal_errors = calculate_reconstruction_errors(
    X_val_tensor,
    model
)


print()

print("Normal Validation Samples:", len(normal_errors))


print(
    "Average Normal Reconstruction Error:",
    round(normal_errors.mean(), 6)
)


print(
    "Maximum Normal Reconstruction Error:",
    round(normal_errors.max(), 6)
)


# ============================================================
# SECTION 5 : CREATE ANOMALY THRESHOLD
# ============================================================

# We use the 95th percentile of normal reconstruction errors.
#
# This means approximately 95% of normal validation samples
# should fall below the threshold.

THRESHOLD_PERCENTILE = 95


threshold = np.percentile(
    normal_errors,
    THRESHOLD_PERCENTILE
)


print()

print("=" * 60)

print("ANOMALY THRESHOLD")

print("=" * 60)


print()

print(
    "Threshold Percentile:",
    THRESHOLD_PERCENTILE
)


print(
    "Anomaly Threshold:",
    round(threshold, 6)
)


# ============================================================
# SECTION 6 : CALCULATE ANOMALY RECONSTRUCTION ERRORS
# ============================================================

print()

print("Calculating Anomaly Reconstruction Errors...")


anomaly_errors = calculate_reconstruction_errors(
    X_anomaly_tensor,
    model
)


print()

print("Anomaly Samples:", len(anomaly_errors))


print(
    "Average Anomaly Reconstruction Error:",
    round(anomaly_errors.mean(), 6)
)


print(
    "Maximum Anomaly Reconstruction Error:",
    round(anomaly_errors.max(), 6)
)


# ============================================================
# SECTION 7 : COMPARE NORMAL AND ANOMALY ERRORS
# ============================================================

print()

print("=" * 60)

print("RECONSTRUCTION ERROR COMPARISON")

print("=" * 60)


print()

print(
    "Normal Average Error:",
    round(normal_errors.mean(), 6)
)


print(
    "Anomaly Average Error:",
    round(anomaly_errors.mean(), 6)
)


print()

print(
    "Anomalies should generally have higher reconstruction errors."
)


# ============================================================
# SECTION 8 : VISUALIZE RECONSTRUCTION ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    normal_errors,
    bins=30,
    alpha=0.6,
    label="Normal"
)


plt.hist(
    anomaly_errors,
    bins=30,
    alpha=0.6,
    label="Anomaly"
)


plt.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="Anomaly Threshold"
)


plt.title(
    "Normal vs Anomaly Reconstruction Error"
)


plt.xlabel(
    "Reconstruction Error"
)


plt.ylabel(
    "Number of Samples"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.show()


# ============================================================
# SECTION 9 : CREATE COMBINED TEST DATA
# ============================================================

# Combine:
#
# Normal Validation Data -> Label 0
# Anomaly Data           -> Label 1


all_errors = np.concatenate(
    [
        normal_errors,
        anomaly_errors
    ]
)


true_labels = np.concatenate(
    [
        np.zeros(
            len(normal_errors),
            dtype=int
        ),

        np.ones(
            len(anomaly_errors),
            dtype=int
        )
    ]
)


print()

print("Combined Evaluation Samples:", len(all_errors))


# ============================================================
# SECTION 10 : DETECT ANOMALIES
# ============================================================

# If reconstruction error is greater than threshold:
#
# 1 -> Anomaly
#
# Otherwise:
#
# 0 -> Normal


predicted_labels = (
    all_errors > threshold
).astype(
    int
)


print()

print("Anomaly Detection Completed Successfully")


# ============================================================
# SECTION 11 : MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


print()

print("=" * 70)

print("ANOMALY DETECTION PERFORMANCE")

print("=" * 70)


print()

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)


print(
    f"Precision : {precision * 100:.2f}%"
)


print(
    f"Recall    : {recall * 100:.2f}%"
)


print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# SECTION 12 : CLASSIFICATION REPORT
# ============================================================

print()

print("=" * 70)

print("CLASSIFICATION REPORT")

print("=" * 70)


print()

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "Normal",
            "Anomaly"
        ],
        zero_division=0
    )
)


# ============================================================
# SECTION 13 : CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


print()

print("=" * 70)

print("CONFUSION MATRIX")

print("=" * 70)


print()

print(cm)


# ============================================================
# SECTION 14 : VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    cm
)


plt.colorbar()


plt.xticks(
    [0, 1],
    [
        "Normal",
        "Anomaly"
    ]
)


plt.yticks(
    [0, 1],
    [
        "Normal",
        "Anomaly"
    ]
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


plt.title(
    "Anomaly Detection Confusion Matrix"
)


# Add values inside the confusion matrix.

for i in range(2):


    for j in range(2):


        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.show()


# ============================================================
# SECTION 15 : VISUALIZE SAMPLE RECONSTRUCTION ERRORS
# ============================================================

plt.figure(
    figsize=(12, 5)
)


# Normal samples.

plt.scatter(
    range(len(normal_errors)),
    normal_errors,
    label="Normal Samples"
)


# Anomaly samples.

plt.scatter(
    range(
        len(normal_errors),
        len(normal_errors) + len(anomaly_errors)
    ),
    anomaly_errors,
    label="Anomaly Samples"
)


# Threshold line.

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)


plt.title(
    "Reconstruction Errors for Normal and Anomaly Samples"
)


plt.xlabel(
    "Sample Index"
)


plt.ylabel(
    "Reconstruction Error"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.show()


# ============================================================
# SECTION 16 : CREATE INDIVIDUAL PREDICTION FUNCTION
# ============================================================

def predict_anomaly(
    data_tensor,
    index,
    actual_label_name
):


    # Put model in evaluation mode.

    model.eval()


    # Select one sample.

    sample = data_tensor[index].unsqueeze(
        0
    ).to(
        device
    )


    # Reconstruct the sample.

    with torch.no_grad():


        reconstructed = model(
            sample
        )


    # Calculate reconstruction error.

    error = torch.mean(
        (
            sample - reconstructed
        ) ** 2
    ).item()


    # Detect anomaly.

    if error > threshold:


        prediction = "Anomaly"


    else:


        prediction = "Normal"


    print()

    print("=" * 60)

    print("INDIVIDUAL ANOMALY PREDICTION")

    print("=" * 60)


    print()

    print(
        "Actual Category:",
        actual_label_name
    )


    print(
        "Predicted Category:",
        prediction
    )


    print(
        "Reconstruction Error:",
        round(error, 6)
    )


    print(
        "Threshold:",
        round(threshold, 6)
    )


    return prediction, error


# ============================================================
# SECTION 17 : TEST NORMAL SAMPLE
# ============================================================

print()

print("TESTING NORMAL SAMPLE")


predict_anomaly(
    X_val_tensor,
    index=0,
    actual_label_name="Normal"
)


# ============================================================
# SECTION 18 : TEST ANOMALY SAMPLE
# ============================================================

print()

print("TESTING ANOMALY SAMPLE")


predict_anomaly(
    X_anomaly_tensor,
    index=0,
    actual_label_name="Anomaly"
)


# ============================================================
# SECTION 19 : SAVE FINAL MODEL INFORMATION
# ============================================================

final_model_info = {

    "project":
        "Anomaly Detection using Autoencoder",

    "dataset":
        "Breast Cancer Wisconsin Dataset",

    "input_features":
        INPUT_DIM,

    "hidden_dimension":
        HIDDEN_DIM,

    "latent_dimension":
        LATENT_DIM,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "threshold_percentile":
        THRESHOLD_PERCENTILE,

    "anomaly_threshold":
        float(threshold),

    "accuracy":
        float(accuracy)

}


print()

print("=" * 70)

print("FINAL MODEL INFORMATION")

print("=" * 70)


print()


for key, value in final_model_info.items():


    print(
        key,
        ":",
        value
    )


# ============================================================
# SECTION 20 : FINAL PROJECT SUMMARY
# ============================================================

print()

print("=" * 70)

print("ANOMALY DETECTION PROJECT COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("PROJECT PIPELINE")


print()

print("✔ Built-in Dataset Loading")

print("✔ Exploratory Data Analysis")

print("✔ Normal Data Selection")

print("✔ Anomaly Data Preparation")

print("✔ Feature Scaling")

print("✔ PyTorch DataLoaders")

print("✔ Autoencoder Development")

print("✔ CPU-Friendly Training")

print("✔ Reconstruction Error Calculation")

print("✔ Automatic Threshold Creation")

print("✔ Anomaly Detection")

print("✔ Model Evaluation")

print("✔ Confusion Matrix")

print("✔ Individual Predictions")

print("✔ Model Saving")


print()

print("MODEL PATH:")

print(
    MODEL_PATH
)


print()

print("END-TO-END ANOMALY DETECTION USING AUTOENCODERS COMPLETE")
