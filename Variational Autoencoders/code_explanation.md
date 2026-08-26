# ============================================================
# PART-1 : DATASET LOADING & EXPLORATORY DATA ANALYSIS
# PROJECT : DATA GENERATION USING VARIATIONAL AUTOENCODERS
# FRAMEWORK : PYTORCH
# DATASET : SKLEARN DIGITS DATASET
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits


print("Libraries Imported Successfully")


# ============================================================
# SECTION 2 : LOAD BUILT-IN DIGITS DATASET
# ============================================================

# The Digits dataset contains small 8x8 grayscale
# images of handwritten digits from 0 to 9.

digits = load_digits()


print()

print("Digits Dataset Loaded Successfully")


# ============================================================
# SECTION 3 : EXTRACT DATA AND LABELS
# ============================================================

# Extract image pixel data.
#
# Shape:
# (number_of_samples, 64)

X = digits.data


# Extract digit labels.
#
# Labels range from 0 to 9.

y = digits.target


print()

print("Data Extracted Successfully")


# ============================================================
# SECTION 4 : DATASET BASIC INFORMATION
# ============================================================

print()

print("=" * 70)

print("DATASET BASIC INFORMATION")

print("=" * 70)


print()

print(

    "Total Number of Images:",

    X.shape[0]

)


print(

    "Number of Features:",

    X.shape[1]

)


print(

    "Original Image Shape:",

    digits.images[0].shape

)


print(

    "Number of Classes:",

    len(np.unique(y))

)


print(

    "Classes:",

    np.unique(y)

)


# ============================================================
# SECTION 5 : CREATE DATAFRAME
# ============================================================

# Convert the 64 pixel features into a Pandas DataFrame.

df = pd.DataFrame(

    X,

    columns=[

        f"pixel_{i}"

        for i in range(

            X.shape[1]

        )

    ]

)


# Add the digit label.

df["label"] = y


print()

print("DataFrame Created Successfully")


# ============================================================
# SECTION 6 : DISPLAY FIRST 5 ROWS
# ============================================================

print()

print("=" * 70)

print("FIRST 5 ROWS OF THE DATASET")

print("=" * 70)


display(

    df.head()

)


# ============================================================
# SECTION 7 : CHECK DATA TYPES
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
# SECTION 8 : CHECK MISSING VALUES
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


print()

print(

    "Total Missing Values:",

    missing_values.sum()

)


# ============================================================
# SECTION 9 : STATISTICAL SUMMARY
# ============================================================

print()

print("=" * 70)

print("STATISTICAL SUMMARY")

print("=" * 70)


display(

    df.describe()

)


# ============================================================
# SECTION 10 : DIGIT CLASS DISTRIBUTION
# ============================================================

print()

print("=" * 70)

print("DIGIT CLASS DISTRIBUTION")

print("=" * 70)


class_distribution = pd.Series(

    y

).value_counts().sort_index()


print()

print(

    class_distribution

)


# ============================================================
# SECTION 11 : VISUALIZE CLASS DISTRIBUTION
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.bar(

    class_distribution.index,

    class_distribution.values

)


plt.title(

    "Digit Class Distribution"

)


plt.xlabel(

    "Digit"

)


plt.ylabel(

    "Number of Images"

)


plt.xticks(

    range(10)

)


plt.grid(

    axis="y",

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 12 : VISUALIZE SINGLE DIGIT IMAGE
# ============================================================

sample_index = 0


sample_image = digits.images[

    sample_index

]


sample_label = y[

    sample_index

]


plt.figure(

    figsize=(4, 4)

)


plt.imshow(

    sample_image,

    cmap="gray"

)


plt.title(

    f"Sample Digit: {sample_label}"

)


plt.axis(

    "off"

)


plt.show()


# ============================================================
# SECTION 13 : VISUALIZE MULTIPLE SAMPLE DIGITS
# ============================================================

# Display one example from each digit class.

fig, axes = plt.subplots(

    2,

    5,

    figsize=(10, 5)

)


for digit in range(10):


    # Find the first image belonging to this digit.

    index = np.where(

        y == digit

    )[0][0]


    # Get the image.

    image = digits.images[index]


    # Select subplot.

    ax = axes.flat[digit]


    # Display image.

    ax.imshow(

        image,

        cmap="gray"

    )


    # Add digit label.

    ax.set_title(

        f"Digit: {digit}"

    )


    # Remove axis.

    ax.axis(

        "off"

    )


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 14 : CHECK PIXEL VALUE RANGE
# ============================================================

print()

print("=" * 70)

print("PIXEL VALUE RANGE")

print("=" * 70)


print()

print(

    "Minimum Pixel Value:",

    X.min()

)


print(

    "Maximum Pixel Value:",

    X.max()

)


print()

print(

    "The pixel values will be normalized in Part-2."

)


# ============================================================
# SECTION 15 : IMAGE DIMENSION ANALYSIS
# ============================================================

IMAGE_HEIGHT = digits.images.shape[1]

IMAGE_WIDTH = digits.images.shape[2]

INPUT_DIM = X.shape[1]


print()

print("=" * 70)

print("IMAGE DIMENSION INFORMATION")

print("=" * 70)


print()

print(

    "Image Height:",

    IMAGE_HEIGHT

)


print(

    "Image Width:",

    IMAGE_WIDTH

)


print(

    "Flattened Input Dimension:",

    INPUT_DIM

)


# ============================================================
# SECTION 16 : DISPLAY RAW PIXEL VALUES
# ============================================================

print()

print("=" * 70)

print("RAW PIXEL VALUES FOR FIRST IMAGE")

print("=" * 70)


print()

print(

    X[0].reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )

)


# ============================================================
# SECTION 17 : FINAL DATASET SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-1 DATASET SUMMARY")

print("=" * 70)


print()

print(

    "Dataset Name: sklearn Digits Dataset"

)


print(

    "Total Images:",

    X.shape[0]

)


print(

    "Image Shape:",

    f"{IMAGE_HEIGHT} x {IMAGE_WIDTH}"

)


print(

    "Input Dimension:",

    INPUT_DIM

)


print(

    "Number of Digit Classes:",

    len(

        np.unique(y)

    )

)


print(

    "Missing Values:",

    missing_values.sum()

)


print(

    "Pixel Value Range:",

    f"{X.min()} to {X.max()}"

)


# ============================================================
# SECTION 18 : PART-1 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-1 COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("Important Variables Created:")

print()

print("✔ digits")

print("✔ X")

print("✔ y")

print("✔ df")

print("✔ IMAGE_HEIGHT")

print("✔ IMAGE_WIDTH")

print("✔ INPUT_DIM")

print("✔ class_distribution")


print()

print("READY FOR PART-2")

print("DATA PREPROCESSING & PYTORCH DATALOADER PREPARATION")
# ============================================================
# PART-2 : DATA PREPROCESSING & PYTORCH DATALOADER PREPARATION
# PROJECT : DATA GENERATION USING VARIATIONAL AUTOENCODERS
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np

import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader


print("Part-2 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : SET RANDOM SEED
# ============================================================

# Setting random seeds helps produce more consistent results.

RANDOM_STATE = 42


np.random.seed(

    RANDOM_STATE

)


torch.manual_seed(

    RANDOM_STATE

)


print()

print("Random Seed Set Successfully")


# ============================================================
# SECTION 3 : CHECK ORIGINAL PIXEL VALUES
# ============================================================

print()

print("=" * 70)

print("ORIGINAL PIXEL VALUE RANGE")

print("=" * 70)


print()

print(

    "Minimum Pixel Value:",

    X.min()

)


print(

    "Maximum Pixel Value:",

    X.max()

)


# ============================================================
# SECTION 4 : NORMALIZE PIXEL VALUES
# ============================================================

# The sklearn Digits dataset has pixel values
# ranging from 0 to 16.
#
# We divide by 16 so that all values are
# between 0 and 1.
#
# Normalized Pixel Value =
#
# Original Pixel Value / 16


X_normalized = X.astype(

    np.float32

) / 16.0


print()

print("Pixel Normalization Completed Successfully")


# ============================================================
# SECTION 5 : CHECK NORMALIZED PIXEL VALUES
# ============================================================

print()

print("=" * 70)

print("NORMALIZED PIXEL VALUE RANGE")

print("=" * 70)


print()

print(

    "Minimum Pixel Value:",

    X_normalized.min()

)


print(

    "Maximum Pixel Value:",

    X_normalized.max()

)


# ============================================================
# SECTION 6 : SPLIT TRAINING AND VALIDATION DATA
# ============================================================

# We split the dataset into:
#
# 80% -> Training Data
# 20% -> Validation Data
#
# The labels are included only for tracking and
# visualization. The VAE learns from the images.


X_train, X_val, y_train, y_val = train_test_split(

    X_normalized,

    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    shuffle=True,

    stratify=y

)


print()

print("=" * 70)

print("DATASET SPLIT")

print("=" * 70)


print()

print(

    "Training Samples:",

    X_train.shape[0]

)


print(

    "Validation Samples:",

    X_val.shape[0]

)


print(

    "Input Features:",

    X_train.shape[1]

)


# ============================================================
# SECTION 7 : CONVERT NUMPY ARRAYS TO PYTORCH TENSORS
# ============================================================

# Convert normalized training data into PyTorch tensors.

X_train_tensor = torch.tensor(

    X_train,

    dtype=torch.float32

)


# Convert normalized validation data into PyTorch tensors.

X_val_tensor = torch.tensor(

    X_val,

    dtype=torch.float32

)


# Labels are not used as VAE targets,
# but we keep them for visualization and analysis.

y_train_tensor = torch.tensor(

    y_train,

    dtype=torch.long

)


y_val_tensor = torch.tensor(

    y_val,

    dtype=torch.long

)


print()

print("PyTorch Tensor Conversion Completed Successfully")


# ============================================================
# SECTION 8 : CHECK TENSOR INFORMATION
# ============================================================

print()

print("=" * 70)

print("PYTORCH TENSOR INFORMATION")

print("=" * 70)


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

    "Training Labels Shape:",

    y_train_tensor.shape

)


print(

    "Validation Labels Shape:",

    y_val_tensor.shape

)


print()

print(

    "Tensor Data Type:",

    X_train_tensor.dtype

)


# ============================================================
# SECTION 9 : CREATE PYTORCH DATASETS
# ============================================================

# For a VAE, the input image is also the
# reconstruction target.
#
# Input:
# Image
#
# Target:
# Same Image


train_dataset = TensorDataset(

    X_train_tensor,

    X_train_tensor,

    y_train_tensor

)


val_dataset = TensorDataset(

    X_val_tensor,

    X_val_tensor,

    y_val_tensor

)


print()

print("PyTorch Datasets Created Successfully")


# ============================================================
# SECTION 10 : SET CPU-FRIENDLY BATCH SIZE
# ============================================================

# Small batch size for slow CPU and limited memory.

BATCH_SIZE = 32


print()

print(

    "Batch Size:",

    BATCH_SIZE

)


# ============================================================
# SECTION 11 : CREATE TRAINING DATALOADER
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


# ============================================================
# SECTION 12 : CREATE VALIDATION DATALOADER
# ============================================================

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


print()

print("DataLoaders Created Successfully")


# ============================================================
# SECTION 13 : CHECK DATALOADER INFORMATION
# ============================================================

print()

print("=" * 70)

print("DATALOADER INFORMATION")

print("=" * 70)


print()

print(

    "Training Batches:",

    len(train_loader)

)


print(

    "Validation Batches:",

    len(val_loader)

)


print(

    "Batch Size:",

    BATCH_SIZE

)


# ============================================================
# SECTION 14 : INSPECT ONE TRAINING BATCH
# ============================================================

sample_images, sample_targets, sample_labels = next(

    iter(train_loader)

)


print()

print("=" * 70)

print("SAMPLE BATCH INFORMATION")

print("=" * 70)


print()

print(

    "Input Images Shape:",

    sample_images.shape

)


print(

    "Target Images Shape:",

    sample_targets.shape

)


print(

    "Labels Shape:",

    sample_labels.shape

)


print()

print(

    "Input and Target Are Equal:",

    torch.equal(

        sample_images,

        sample_targets

    )

)


print()

print(

    "Sample Labels:",

    sample_labels[:10].numpy()

)


# ============================================================
# SECTION 15 : CHECK COMPUTATION DEVICE
# ============================================================

# Check whether CUDA GPU is available.
#
# If not, CPU will be used.

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print()

print("=" * 70)

print("COMPUTATION DEVICE")

print("=" * 70)


print()

print(

    "Using Device:",

    device

)


# ============================================================
# SECTION 16 : VISUALIZE NORMALIZED IMAGES
# ============================================================

# Display a few images after normalization.

fig, axes = plt.subplots(

    2,

    5,

    figsize=(10, 5)

)


for index in range(10):


    image = X_train_tensor[index].reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    ax = axes.flat[index]


    ax.imshow(

        image,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    ax.set_title(

        f"Digit: {y_train[index]}"

    )


    ax.axis(

        "off"

    )


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 17 : DEFINE FINAL INPUT DIMENSION
# ============================================================

# Input dimension is the number of flattened pixels.

INPUT_DIM = X_train_tensor.shape[1]


print()

print("=" * 70)

print("FINAL MODEL INPUT INFORMATION")

print("=" * 70)


print()

print(

    "Input Dimension:",

    INPUT_DIM

)


print(

    "Image Height:",

    IMAGE_HEIGHT

)


print(

    "Image Width:",

    IMAGE_WIDTH

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

    "Total Dataset Samples:",

    len(X_normalized)

)


print(

    "Training Samples:",

    len(X_train_tensor)

)


print(

    "Validation Samples:",

    len(X_val_tensor)

)


print(

    "Input Dimension:",

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

print("✔ X_normalized")

print("✔ X_train")

print("✔ X_val")

print("✔ y_train")

print("✔ y_val")

print("✔ X_train_tensor")

print("✔ X_val_tensor")

print("✔ y_train_tensor")

print("✔ y_val_tensor")

print("✔ train_dataset")

print("✔ val_dataset")

print("✔ train_loader")

print("✔ val_loader")

print("✔ BATCH_SIZE")

print("✔ INPUT_DIM")

print("✔ device")


print()

print("READY FOR PART-3")

print("VARIATIONAL AUTOENCODER DEVELOPMENT & TRAINING")
# ============================================================
# PART-3 : VARIATIONAL AUTOENCODER DEVELOPMENT & TRAINING
# PROJECT : DATA GENERATION USING VARIATIONAL AUTOENCODERS
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np

import torch

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt


print("Part-3 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : SET VAE HYPERPARAMETERS
# ============================================================

# Small architecture for CPU-friendly training.

HIDDEN_DIM = 64

LATENT_DIM = 8

LEARNING_RATE = 0.001

EPOCHS = 40


print()

print("=" * 70)

print("VAE CONFIGURATION")

print("=" * 70)


print()

print("Input Dimension:", INPUT_DIM)

print("Hidden Dimension:", HIDDEN_DIM)

print("Latent Dimension:", LATENT_DIM)

print("Learning Rate:", LEARNING_RATE)

print("Epochs:", EPOCHS)


# ============================================================
# SECTION 3 : CREATE VARIATIONAL AUTOENCODER CLASS
# ============================================================

class VariationalAutoencoder(nn.Module):


    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim
    ):


        # Initialize the parent PyTorch module.

        super().__init__()


        # ====================================================
        # ENCODER
        # ====================================================
        #
        # Input
        # 64 Features
        #      ↓
        # Hidden Layer
        # 64 Neurons
        #

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU()

        )


        # ====================================================
        # LATENT MEAN
        # ====================================================
        #
        # Generates the mean (μ) of the
        # latent probability distribution.

        self.fc_mu = nn.Linear(
            hidden_dim,
            latent_dim
        )


        # ====================================================
        # LATENT LOG VARIANCE
        # ====================================================
        #
        # Generates log(σ²).

        self.fc_logvar = nn.Linear(
            hidden_dim,
            latent_dim
        )


        # ====================================================
        # DECODER
        # ====================================================
        #
        # Latent Vector
        # 8 Features
        #      ↓
        # Hidden Layer
        # 64 Neurons
        #      ↓
        # Output
        # 64 Pixels
        #
        # Sigmoid keeps output values
        # between 0 and 1.

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                input_dim
            ),

            nn.Sigmoid()

        )


    # ========================================================
    # ENCODE FUNCTION
    # ========================================================

    def encode(
        self,
        x
    ):


        # Pass input through encoder.

        hidden = self.encoder(
            x
        )


        # Generate latent mean.

        mu = self.fc_mu(
            hidden
        )


        # Generate latent log variance.

        logvar = self.fc_logvar(
            hidden
        )


        return mu, logvar


    # ========================================================
    # REPARAMETERIZATION TRICK
    # ========================================================

    def reparameterize(
        self,
        mu,
        logvar
    ):


        # Convert log variance into
        # standard deviation.
        #
        # std = exp(0.5 * logvar)

        std = torch.exp(
            0.5 * logvar
        )


        # Generate random noise from
        # standard normal distribution.

        epsilon = torch.randn_like(
            std
        )


        # Sample latent vector.
        #
        # z = μ + σ * ε

        z = mu + (
            std * epsilon
        )


        return z


    # ========================================================
    # DECODE FUNCTION
    # ========================================================

    def decode(
        self,
        z
    ):


        # Convert latent vector back
        # into reconstructed image.

        reconstruction = self.decoder(
            z
        )


        return reconstruction


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(
        self,
        x
    ):


        # Step 1:
        # Encode the input.

        mu, logvar = self.encode(
            x
        )


        # Step 2:
        # Sample latent vector.

        z = self.reparameterize(
            mu,
            logvar
        )


        # Step 3:
        # Reconstruct the input.

        reconstruction = self.decode(
            z
        )


        return reconstruction, mu, logvar


print()

print("Variational Autoencoder Class Created Successfully")


# ============================================================
# SECTION 4 : CREATE VAE MODEL
# ============================================================

model = VariationalAutoencoder(

    input_dim=INPUT_DIM,

    hidden_dim=HIDDEN_DIM,

    latent_dim=LATENT_DIM

)


# Move model to CPU or GPU.

model = model.to(
    device
)


print()

print("=" * 70)

print("VAE MODEL")

print("=" * 70)


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

print(
    "Total Trainable Parameters:",
    total_parameters
)


# ============================================================
# SECTION 6 : DEFINE VAE LOSS FUNCTION
# ============================================================

def vae_loss(
    reconstruction,
    original,
    mu,
    logvar
):


    # ========================================================
    # RECONSTRUCTION LOSS
    # ========================================================
    #
    # Measures the difference between:
    #
    # Original Image
    #
    # and
    #
    # Reconstructed Image

    reconstruction_loss = nn.functional.binary_cross_entropy(

        reconstruction,

        original,

        reduction="sum"

    )


    # ========================================================
    # KL DIVERGENCE LOSS
    # ========================================================
    #
    # Forces the latent distribution to
    # stay close to a standard normal
    # distribution.

    kl_divergence = -0.5 * torch.sum(

        1

        +

        logvar

        -

        mu.pow(2)

        -

        logvar.exp()

    )


    # ========================================================
    # TOTAL VAE LOSS
    # ========================================================

    total_loss = (

        reconstruction_loss

        +

        kl_divergence

    )


    return (

        total_loss,

        reconstruction_loss,

        kl_divergence

    )


print()

print("VAE Loss Function Created Successfully")


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

train_reconstruction_losses = []

train_kl_losses = []


# ============================================================
# SECTION 9 : START VAE TRAINING
# ============================================================

print()

print("=" * 70)

print("STARTING VAE TRAINING")

print("=" * 70)


for epoch in range(EPOCHS):


    # ========================================================
    # TRAINING MODE
    # ========================================================

    model.train()


    total_train_loss = 0.0

    total_train_reconstruction_loss = 0.0

    total_train_kl_loss = 0.0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for inputs, targets, labels in train_loader:


        # Move input and target to device.

        inputs = inputs.to(
            device
        )


        targets = targets.to(
            device
        )


        # Clear previous gradients.

        optimizer.zero_grad()


        # Forward pass.

        reconstruction, mu, logvar = model(
            inputs
        )


        # Calculate VAE losses.

        loss, reconstruction_loss, kl_loss = vae_loss(

            reconstruction,

            targets,

            mu,

            logvar

        )


        # Backpropagation.

        loss.backward()


        # Update model parameters.

        optimizer.step()


        # Store losses.

        total_train_loss += loss.item()

        total_train_reconstruction_loss += (
            reconstruction_loss.item()
        )

        total_train_kl_loss += (
            kl_loss.item()
        )


    # ========================================================
    # CALCULATE AVERAGE TRAINING LOSSES
    # ========================================================

    average_train_loss = (

        total_train_loss

        /

        len(X_train_tensor)

    )


    average_reconstruction_loss = (

        total_train_reconstruction_loss

        /

        len(X_train_tensor)

    )


    average_kl_loss = (

        total_train_kl_loss

        /

        len(X_train_tensor)

    )


    # Save losses.

    train_losses.append(
        average_train_loss
    )


    train_reconstruction_losses.append(
        average_reconstruction_loss
    )


    train_kl_losses.append(
        average_kl_loss
    )


    # ========================================================
    # VALIDATION MODE
    # ========================================================

    model.eval()


    total_val_loss = 0.0


    # Disable gradient calculation.

    with torch.no_grad():


        for inputs, targets, labels in val_loader:


            # Move data to device.

            inputs = inputs.to(
                device
            )


            targets = targets.to(
                device
            )


            # Forward pass.

            reconstruction, mu, logvar = model(
                inputs
            )


            # Calculate validation loss.

            val_loss, _, _ = vae_loss(

                reconstruction,

                targets,

                mu,

                logvar

            )


            total_val_loss += val_loss.item()


    # Calculate average validation loss.

    average_val_loss = (

        total_val_loss

        /

        len(X_val_tensor)

    )


    # Save validation loss.

    val_losses.append(
        average_val_loss
    )


    # ========================================================
    # DISPLAY TRAINING PROGRESS
    # ========================================================

    if (

        (epoch + 1) % 5 == 0

        or

        epoch == 0

    ):


        print(

            f"Epoch [{epoch + 1}/{EPOCHS}] "

            f"| Train Loss: {average_train_loss:.4f} "

            f"| Val Loss: {average_val_loss:.4f} "

            f"| Reconstruction: {average_reconstruction_loss:.4f} "

            f"| KL: {average_kl_loss:.4f}"

        )


# ============================================================
# SECTION 10 : TRAINING COMPLETED
# ============================================================

print()

print("=" * 70)

print("VAE TRAINING COMPLETED")

print("=" * 70)


print()

print(

    "Final Training Loss:",

    round(
        train_losses[-1],
        4
    )

)


print(

    "Final Validation Loss:",

    round(
        val_losses[-1],
        4
    )

)


print(

    "Final Reconstruction Loss:",

    round(
        train_reconstruction_losses[-1],
        4
    )

)


print(

    "Final KL Divergence Loss:",

    round(
        train_kl_losses[-1],
        4
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
    "VAE Training and Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Average Loss per Sample"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.show()


# ============================================================
# SECTION 12 : VISUALIZE LOSS COMPONENTS
# ============================================================

plt.figure(
    figsize=(10, 5)
)


plt.plot(

    train_reconstruction_losses,

    label="Reconstruction Loss"

)


plt.plot(

    train_kl_losses,

    label="KL Divergence Loss"

)


plt.title(
    "VAE Loss Components"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Average Loss per Sample"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.show()


# ============================================================
# SECTION 13 : TEST SAMPLE RECONSTRUCTION
# ============================================================

# Put model into evaluation mode.

model.eval()


# Select one validation image.

sample_input = X_val_tensor[0].unsqueeze(
    0
).to(
    device
)


with torch.no_grad():


    reconstruction, mu, logvar = model(
        sample_input
    )


# Convert tensors to NumPy arrays.

original_image = (

    sample_input

    .cpu()

    .numpy()

    .reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )

)


reconstructed_image = (

    reconstruction

    .cpu()

    .numpy()

    .reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )

)


# ============================================================
# SECTION 14 : VISUALIZE ORIGINAL VS RECONSTRUCTED IMAGE
# ============================================================

fig, axes = plt.subplots(

    1,

    2,

    figsize=(8, 4)

)


# Original image.

axes[0].imshow(

    original_image,

    cmap="gray",

    vmin=0,

    vmax=1

)


axes[0].set_title(

    f"Original Digit: {y_val[0]}"

)


axes[0].axis(
    "off"
)


# Reconstructed image.

axes[1].imshow(

    reconstructed_image,

    cmap="gray",

    vmin=0,

    vmax=1

)


axes[1].set_title(
    "VAE Reconstruction"
)


axes[1].axis(
    "off"
)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 15 : SAVE TRAINED VAE MODEL
# ============================================================

MODEL_PATH = "variational_autoencoder_digits.pth"


torch.save(

    model.state_dict(),

    MODEL_PATH

)


print()

print("VAE Model Saved Successfully")

print(
    "Model Path:",
    MODEL_PATH
)


# ============================================================
# SECTION 16 : PART-3 SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-3 COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("Completed:")

print()

print("✔ VAE Encoder")

print("✔ Latent Mean (mu)")

print("✔ Latent Log Variance (logvar)")

print("✔ Reparameterization Trick")

print("✔ VAE Decoder")

print("✔ Reconstruction Loss")

print("✔ KL Divergence Loss")

print("✔ VAE Training")

print("✔ Validation")

print("✔ Loss Visualization")

print("✔ Original vs Reconstructed Image")

print("✔ Model Saving")


print()

print("READY FOR PART-4")

print("GENERATING NEW SYNTHETIC DIGITS USING THE VAE")
# ============================================================
# PART-4 : SYNTHETIC DATA GENERATION & FINAL EVALUATION
# PROJECT : DATA GENERATION USING VARIATIONAL AUTOENCODERS
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np

import torch

import matplotlib.pyplot as plt


print("Part-4 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : SET MODEL TO EVALUATION MODE
# ============================================================

# We already trained the model in Part-3.
# Evaluation mode is used for generation and testing.

model.eval()


print()

print("Trained VAE Ready for Synthetic Data Generation")


# ============================================================
# SECTION 3 : GENERATE RANDOM LATENT VECTORS
# ============================================================

# A VAE learns a latent space that approximately follows
# a standard normal distribution.
#
# Therefore, we generate random values from:
#
# N(0, 1)

NUM_SAMPLES = 10


random_latent_vectors = torch.randn(

    NUM_SAMPLES,

    LATENT_DIM

).to(

    device

)


print()

print("=" * 70)

print("RANDOM LATENT VECTORS CREATED")

print("=" * 70)


print()

print(

    "Latent Vector Shape:",

    random_latent_vectors.shape

)


# ============================================================
# SECTION 4 : GENERATE SYNTHETIC DATA
# ============================================================

# The decoder converts random latent vectors
# into completely new digit images.

with torch.no_grad():


    generated_images = model.decode(

        random_latent_vectors

    )


print()

print("Synthetic Digit Data Generated Successfully")


print()

print(

    "Generated Data Shape:",

    generated_images.shape

)


# ============================================================
# SECTION 5 : VISUALIZE GENERATED DIGITS
# ============================================================

fig, axes = plt.subplots(

    2,

    5,

    figsize=(10, 5)

)


for index in range(NUM_SAMPLES):


    # Convert the flattened output back into 8x8 format.

    image = generated_images[

        index

    ].cpu().numpy().reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    # Display generated image.

    ax = axes.flat[index]


    ax.imshow(

        image,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    ax.set_title(

        f"Generated {index + 1}"

    )


    ax.axis(

        "off"

    )


plt.suptitle(

    "Synthetic Digits Generated by VAE",

    fontsize=14

)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 6 : GENERATE MORE SYNTHETIC SAMPLES
# ============================================================

NUM_GENERATED_IMAGES = 20


random_latent_vectors = torch.randn(

    NUM_GENERATED_IMAGES,

    LATENT_DIM

).to(

    device

)


with torch.no_grad():


    more_generated_images = model.decode(

        random_latent_vectors

    )


fig, axes = plt.subplots(

    4,

    5,

    figsize=(10, 8)

)


for index in range(NUM_GENERATED_IMAGES):


    image = more_generated_images[

        index

    ].cpu().numpy().reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    ax = axes.flat[index]


    ax.imshow(

        image,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    ax.set_title(

        f"Sample {index + 1}"

    )


    ax.axis(

        "off"

    )


plt.suptitle(

    "20 New Synthetic Digits",

    fontsize=14

)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 7 : CALCULATE VALIDATION RECONSTRUCTION LOSS
# ============================================================

model.eval()


total_reconstruction_error = 0.0

total_samples = 0


with torch.no_grad():


    for inputs, targets, labels in val_loader:


        # Move input data to device.

        inputs = inputs.to(

            device

        )


        targets = targets.to(

            device

        )


        # Reconstruct validation images.

        reconstruction, mu, logvar = model(

            inputs

        )


        # Calculate Mean Squared Error.

        reconstruction_error = torch.sum(

            (

                reconstruction

                -

                targets

            ) ** 2

        )


        # Add error.

        total_reconstruction_error += (

            reconstruction_error.item()

        )


        # Count samples.

        total_samples += (

            inputs.size(

                0

            )

        )


average_reconstruction_error = (

    total_reconstruction_error

    /

    total_samples

)


print()

print("=" * 70)

print("VALIDATION RECONSTRUCTION EVALUATION")

print("=" * 70)


print()

print(

    "Average Reconstruction Error per Sample:",

    round(

        average_reconstruction_error,

        6

    )

)


# ============================================================
# SECTION 8 : COMPARE MULTIPLE ORIGINAL AND RECONSTRUCTED IMAGES
# ============================================================

NUM_COMPARISON_IMAGES = 5


model.eval()


with torch.no_grad():


    comparison_input = X_val_tensor[

        :NUM_COMPARISON_IMAGES

    ].to(

        device

    )


    comparison_reconstruction, _, _ = model(

        comparison_input

    )


fig, axes = plt.subplots(

    2,

    NUM_COMPARISON_IMAGES,

    figsize=(12, 5)

)


for index in range(NUM_COMPARISON_IMAGES):


    # Original image.

    original = comparison_input[

        index

    ].cpu().numpy().reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    axes[0, index].imshow(

        original,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    axes[0, index].set_title(

        f"Original: {y_val[index]}"

    )


    axes[0, index].axis(

        "off"

    )


    # Reconstructed image.

    reconstructed = comparison_reconstruction[

        index

    ].cpu().numpy().reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    axes[1, index].imshow(

        reconstructed,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    axes[1, index].set_title(

        "Reconstructed"

    )


    axes[1, index].axis(

        "off"

    )


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 9 : EXTRACT LATENT REPRESENTATIONS
# ============================================================

# We use the latent mean (mu) as the deterministic
# representation of each validation image.

model.eval()


latent_vectors = []

latent_labels = []


with torch.no_grad():


    for inputs, targets, labels in val_loader:


        inputs = inputs.to(

            device

        )


        # Encode input.

        mu, logvar = model.encode(

            inputs

        )


        # Store latent mean.

        latent_vectors.append(

            mu.cpu().numpy()

        )


        # Store digit labels.

        latent_labels.append(

            labels.numpy()

        )


# Combine all batches.

latent_vectors = np.concatenate(

    latent_vectors,

    axis=0

)


latent_labels = np.concatenate(

    latent_labels,

    axis=0

)


print()

print("=" * 70)

print("LATENT SPACE INFORMATION")

print("=" * 70)


print()

print(

    "Latent Space Shape:",

    latent_vectors.shape

)


# ============================================================
# SECTION 10 : VISUALIZE 2 DIMENSIONS OF LATENT SPACE
# ============================================================

# LATENT_DIM = 8.
#
# For simple visualization, we plot
# the first two dimensions.

plt.figure(

    figsize=(10, 7)

)


scatter = plt.scatter(

    latent_vectors[:, 0],

    latent_vectors[:, 1],

    c=latent_labels,

    alpha=0.7

)


plt.colorbar(

    scatter,

    label="Digit Label"

)


plt.title(

    "VAE Latent Space Visualization"

)


plt.xlabel(

    "Latent Dimension 1"

)


plt.ylabel(

    "Latent Dimension 2"

)


plt.grid(

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 11 : LATENT SPACE INTERPOLATION
# ============================================================

# Select two validation samples.
#
# We encode both samples into the latent space
# and gradually move from one latent representation
# to the other.

sample_1_index = 0

sample_2_index = 1


model.eval()


with torch.no_grad():


    sample_1 = X_val_tensor[

        sample_1_index

    ].unsqueeze(

        0

    ).to(

        device

    )


    sample_2 = X_val_tensor[

        sample_2_index

    ].unsqueeze(

        0

    ).to(

        device

    )


    # Get latent means.

    mu_1, _ = model.encode(

        sample_1

    )


    mu_2, _ = model.encode(

        sample_2

    )


# Number of interpolation steps.

NUM_STEPS = 8


interpolation_values = torch.linspace(

    0,

    1,

    NUM_STEPS

).to(

    device

)


interpolated_images = []


with torch.no_grad():


    for alpha in interpolation_values:


        # Linear interpolation.
        #
        # z = (1 - alpha) * z1 + alpha * z2

        interpolated_latent = (

            (1 - alpha)

            *

            mu_1

            +

            alpha

            *

            mu_2

        )


        # Decode latent vector.

        generated_image = model.decode(

            interpolated_latent

        )


        interpolated_images.append(

            generated_image

            .cpu()

            .numpy()

        )


# ============================================================
# SECTION 12 : VISUALIZE LATENT INTERPOLATION
# ============================================================

fig, axes = plt.subplots(

    1,

    NUM_STEPS,

    figsize=(15, 3)

)


for index in range(NUM_STEPS):


    image = interpolated_images[

        index

    ].reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    axes[index].imshow(

        image,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    axes[index].set_title(

        f"{index + 1}"

    )


    axes[index].axis(

        "off"

    )


plt.suptitle(

    f"Latent Space Interpolation: {y_val[sample_1_index]} → {y_val[sample_2_index]}"

)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 13 : CREATE REUSABLE DIGIT GENERATION FUNCTION
# ============================================================

def generate_digits(

    number_of_digits=10

):


    # Put model in evaluation mode.

    model.eval()


    # Generate random latent vectors.

    random_latent = torch.randn(

        number_of_digits,

        LATENT_DIM

    ).to(

        device

    )


    # Generate images.

    with torch.no_grad():


        synthetic_images = model.decode(

            random_latent

        )


    # Move data to CPU.

    synthetic_images = synthetic_images.cpu()


    return synthetic_images


print()

print("Reusable Digit Generation Function Created Successfully")


# ============================================================
# SECTION 14 : TEST GENERATION FUNCTION
# ============================================================

generated_test_digits = generate_digits(

    number_of_digits=5

)


print()

print(

    "Generated Tensor Shape:",

    generated_test_digits.shape

)


# ============================================================
# SECTION 15 : VISUALIZE FUNCTION-GENERATED DIGITS
# ============================================================

fig, axes = plt.subplots(

    1,

    5,

    figsize=(10, 3)

)


for index in range(5):


    image = generated_test_digits[

        index

    ].numpy().reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    axes[index].imshow(

        image,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    axes[index].set_title(

        f"Generated {index + 1}"

    )


    axes[index].axis(

        "off"

    )


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 16 : SAVE FINAL PROJECT INFORMATION
# ============================================================

final_project_info = {


    "project":

        "Data Generation using Variational Autoencoder",


    "dataset":

        "scikit-learn Digits Dataset",


    "total_samples":

        len(X),


    "image_shape":

        f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}",


    "input_dimension":

        INPUT_DIM,


    "hidden_dimension":

        HIDDEN_DIM,


    "latent_dimension":

        LATENT_DIM,


    "batch_size":

        BATCH_SIZE,


    "epochs":

        EPOCHS,


    "final_training_loss":

        float(train_losses[-1]),


    "final_validation_loss":

        float(val_losses[-1]),


    "average_reconstruction_error":

        float(average_reconstruction_error),


    "model_path":

        MODEL_PATH

}


print()

print("=" * 70)

print("FINAL PROJECT INFORMATION")

print("=" * 70)


print()


for key, value in final_project_info.items():


    print(

        key,

        ":",

        value

    )


# ============================================================
# SECTION 17 : FINAL PROJECT SUMMARY
# ============================================================

print()

print("=" * 70)

print("DATA GENERATION USING VARIATIONAL AUTOENCODER")

print("PROJECT COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("COMPLETE PIPELINE")


print()

print("✔ Built-in Digits Dataset")

print("✔ Exploratory Data Analysis")

print("✔ Pixel Normalization")

print("✔ Train / Validation Split")

print("✔ PyTorch DataLoaders")

print("✔ VAE Encoder")

print("✔ Mean and Log Variance")

print("✔ Reparameterization Trick")

print("✔ Latent Space Learning")

print("✔ VAE Decoder")

print("✔ Reconstruction Loss")

print("✔ KL Divergence Loss")

print("✔ CPU-Friendly Training")

print("✔ Validation")

print("✔ Synthetic Data Generation")

print("✔ Original vs Reconstruction Comparison")

print("✔ Latent Space Visualization")

print("✔ Latent Space Interpolation")

print("✔ Reusable Generation Function")

print("✔ Model Saving")


print()

print("MODEL SAVED AT:")

print(

    MODEL_PATH

)


print()

print("END-TO-END VAE DATA GENERATION PROJECT COMPLETE")
