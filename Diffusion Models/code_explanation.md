# ============================================================
# PART-1 : DATASET LOADING & EXPLORATORY DATA ANALYSIS
# PROJECT : AI IMAGE GENERATION USING DIFFUSION MODEL
# FRAMEWORK : PYTORCH
# DATASET : SCIKIT-LEARN DIGITS DATASET
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

# The Digits dataset contains handwritten digits.
#
# Total Images : 1797
# Image Size   : 8 x 8
# Classes      : 0 to 9
# Pixel Range  : 0 to 16

digits = load_digits()


print()

print("Digits Dataset Loaded Successfully")


# ============================================================
# SECTION 3 : EXTRACT IMAGE DATA AND LABELS
# ============================================================

# X contains flattened image data.
#
# Shape:
#
# (1797, 64)
#
# Each image contains:
#
# 8 x 8 = 64 pixels

X = digits.data


# y contains digit labels.

y = digits.target


# images contains the original
# 8 x 8 image representation.

images = digits.images


print()

print("Image Data Extracted Successfully")


# ============================================================
# SECTION 4 : DATASET BASIC INFORMATION
# ============================================================

print()

print("=" * 70)

print("DATASET BASIC INFORMATION")

print("=" * 70)


print()

print("Total Number of Images:", X.shape[0])

print("Number of Pixel Features:", X.shape[1])

print("Original Image Shape:", images[0].shape)

print("Number of Classes:", len(np.unique(y)))

print("Available Classes:", np.unique(y))


# ============================================================
# SECTION 5 : CREATE DATAFRAME
# ============================================================

# Convert flattened pixel data
# into a Pandas DataFrame.

df = pd.DataFrame(

    X,

    columns=[

        f"pixel_{index}"

        for index in range(X.shape[1])

    ]

)


# Add digit labels.

df["label"] = y


print()

print("DataFrame Created Successfully")


# ============================================================
# SECTION 6 : DISPLAY FIRST 5 ROWS
# ============================================================

print()

print("=" * 70)

print("FIRST 5 ROWS OF DATASET")

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
# SECTION 10 : ANALYZE DIGIT DISTRIBUTION
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
# SECTION 11 : VISUALIZE DIGIT DISTRIBUTION
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.bar(

    class_distribution.index,

    class_distribution.values

)


plt.title(

    "Distribution of Digits in Dataset"

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
# SECTION 12 : VISUALIZE ONE SAMPLE IMAGE
# ============================================================

sample_index = 0


sample_image = images[

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

fig, axes = plt.subplots(

    2,

    5,

    figsize=(10, 5)

)


for digit in range(10):


    # Find one example of each digit.

    index = np.where(

        y == digit

    )[0][0]


    # Get the image.

    image = images[

        index

    ]


    # Select subplot.

    ax = axes.flat[

        digit

    ]


    # Display image.

    ax.imshow(

        image,

        cmap="gray"

    )


    ax.set_title(

        f"Digit: {digit}"

    )


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

print("PIXEL VALUE ANALYSIS")

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

    "Pixel values range from",

    X.min(),

    "to",

    X.max()

)


# ============================================================
# SECTION 15 : DEFINE IMAGE DIMENSIONS
# ============================================================

IMAGE_HEIGHT = images.shape[1]

IMAGE_WIDTH = images.shape[2]

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
# SECTION 16 : DISPLAY RAW PIXEL MATRIX
# ============================================================

print()

print("=" * 70)

print("RAW PIXEL MATRIX FOR FIRST IMAGE")

print("=" * 70)


print()


print(

    X[0].reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )

)


# ============================================================
# SECTION 17 : PIXEL INTENSITY DISTRIBUTION
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.hist(

    X.flatten(),

    bins=17

)


plt.title(

    "Distribution of Pixel Intensities"

)


plt.xlabel(

    "Pixel Value"

)


plt.ylabel(

    "Frequency"

)


plt.grid(

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 18 : FINAL DATASET SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-1 DATASET SUMMARY")

print("=" * 70)


print()

print("Dataset Name: sklearn Digits Dataset")

print("Total Images:", X.shape[0])

print("Image Shape:", f"{IMAGE_HEIGHT} x {IMAGE_WIDTH}")

print("Flattened Input Dimension:", INPUT_DIM)

print("Number of Classes:", len(np.unique(y)))

print("Missing Values:", missing_values.sum())

print(

    "Pixel Value Range:",

    f"{X.min()} to {X.max()}"

)


# ============================================================
# SECTION 19 : PART-1 COMPLETION
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

print("✔ images")

print("✔ df")

print("✔ class_distribution")

print("✔ IMAGE_HEIGHT")

print("✔ IMAGE_WIDTH")

print("✔ INPUT_DIM")


print()

print("READY FOR PART-2")

print("DATA PREPROCESSING & DIFFUSION PROCESS SETUP")
# ============================================================
# PART-2 : DATA PREPROCESSING & FORWARD DIFFUSION SETUP
# PROJECT : AI IMAGE GENERATION USING DIFFUSION MODEL
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np

import torch

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt


print("Part-2 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : SET RANDOM SEED
# ============================================================

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
# SECTION 3 : CHECK ORIGINAL DATA
# ============================================================

print()

print("=" * 70)

print("ORIGINAL DATA INFORMATION")

print("=" * 70)


print()

print("Original X Shape:", X.shape)

print("Original Pixel Minimum:", X.min())

print("Original Pixel Maximum:", X.max())


# ============================================================
# SECTION 4 : NORMALIZE PIXELS FROM 0-16 TO 0-1
# ============================================================

# The sklearn Digits dataset has pixel values
# between 0 and 16.
#
# Divide by 16 to convert the range to:
#
# 0 to 1

X_normalized = X.astype(

    np.float32

) / 16.0


print()

print("Normalization from 0-16 to 0-1 Completed")


print()

print("Minimum Value:", X_normalized.min())

print("Maximum Value:", X_normalized.max())


# ============================================================
# SECTION 5 : SCALE PIXELS FROM 0-1 TO -1 TO +1
# ============================================================

# Diffusion models commonly work with
# normalized image values.
#
# Formula:
#
# X_scaled = (X_normalized * 2) - 1
#
# Final Range:
#
# -1 to +1

X_diffusion = (

    X_normalized * 2.0

) - 1.0


print()

print("Scaling from 0-1 to -1,+1 Completed")


print()

print("Minimum Value:", X_diffusion.min())

print("Maximum Value:", X_diffusion.max())


# ============================================================
# SECTION 6 : CONVERT NUMPY ARRAY TO PYTORCH TENSOR
# ============================================================

X_tensor = torch.tensor(

    X_diffusion,

    dtype=torch.float32

)


print()

print("PyTorch Tensor Created Successfully")


print()

print("Tensor Shape:", X_tensor.shape)

print("Tensor Data Type:", X_tensor.dtype)


# ============================================================
# SECTION 7 : CREATE PYTORCH DATASET
# ============================================================

# For unconditional diffusion training,
# we only need images.
#
# Labels are not required.

diffusion_dataset = TensorDataset(

    X_tensor

)


print()

print("Diffusion Dataset Created Successfully")


# ============================================================
# SECTION 8 : SET CPU-FRIENDLY BATCH SIZE
# ============================================================

BATCH_SIZE = 32


print()

print("Batch Size:", BATCH_SIZE)


# ============================================================
# SECTION 9 : CREATE DATALOADER
# ============================================================

train_loader = DataLoader(

    diffusion_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


print()

print("DataLoader Created Successfully")


# ============================================================
# SECTION 10 : CHECK ONE BATCH
# ============================================================

sample_batch = next(

    iter(train_loader)

)


real_images = sample_batch[0]


print()

print("=" * 70)

print("SAMPLE BATCH INFORMATION")

print("=" * 70)


print()

print("Batch Shape:", real_images.shape)

print("Batch Minimum:", real_images.min().item())

print("Batch Maximum:", real_images.max().item())


# ============================================================
# SECTION 11 : SET COMPUTATION DEVICE
# ============================================================

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

print("Using Device:", device)


# ============================================================
# SECTION 12 : DEFINE DIFFUSION CONFIGURATION
# ============================================================

# Number of diffusion steps.
#
# We keep this small because the system
# is CPU-friendly.

TIMESTEPS = 50


# Beta controls how much noise is added
# at each diffusion step.

BETA_START = 0.0001

BETA_END = 0.02


print()

print("=" * 70)

print("DIFFUSION CONFIGURATION")

print("=" * 70)


print()

print("Number of Diffusion Steps:", TIMESTEPS)

print("Beta Start:", BETA_START)

print("Beta End:", BETA_END)


# ============================================================
# SECTION 13 : CREATE BETA NOISE SCHEDULE
# ============================================================

# A linear schedule gradually increases
# the amount of noise added.

betas = torch.linspace(

    BETA_START,

    BETA_END,

    TIMESTEPS

).to(

    device

)


print()

print("Beta Noise Schedule Created Successfully")


# ============================================================
# SECTION 14 : CALCULATE ALPHAS
# ============================================================

# Alpha represents the amount of original
# image information retained at each step.
#
# Formula:
#
# alpha = 1 - beta

alphas = 1.0 - betas


print()

print("Alphas Calculated Successfully")


# ============================================================
# SECTION 15 : CALCULATE CUMULATIVE ALPHAS
# ============================================================

# alpha_hat represents how much original
# information remains after multiple steps.
#
# Formula:
#
# alpha_hat[t] =
#
# alpha[0] * alpha[1] * ... * alpha[t]

alpha_hat = torch.cumprod(

    alphas,

    dim=0

)


print()

print("Cumulative Alphas Calculated Successfully")


# ============================================================
# SECTION 16 : VISUALIZE NOISE SCHEDULE
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.plot(

    betas.cpu().numpy()

)


plt.title(

    "Diffusion Beta Noise Schedule"

)


plt.xlabel(

    "Timestep"

)


plt.ylabel(

    "Beta Value"

)


plt.grid(

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 17 : VISUALIZE ALPHA HAT
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.plot(

    alpha_hat.cpu().numpy()

)


plt.title(

    "Cumulative Alpha Values During Diffusion"

)


plt.xlabel(

    "Timestep"

)


plt.ylabel(

    "Alpha Hat"

)


plt.grid(

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 18 : CREATE FORWARD DIFFUSION FUNCTION
# ============================================================

# Forward Diffusion Formula:
#
# x_t =
#
# sqrt(alpha_hat_t) * x_0
#
# +
#
# sqrt(1 - alpha_hat_t) * noise
#
# Where:
#
# x_0 = Original Image
#
# x_t = Noisy Image
#
# noise = Random Gaussian Noise


def forward_diffusion(

    images,

    timesteps

):


    # Get random Gaussian noise.

    noise = torch.randn_like(

        images

    )


    # Get alpha_hat values for each image.
    #
    # Reshape allows broadcasting across
    # all 64 pixel values.

    alpha_hat_t = alpha_hat[

        timesteps

    ].unsqueeze(

        1

    )


    # Calculate noisy images.

    noisy_images = (

        torch.sqrt(

            alpha_hat_t

        )

        *

        images

        +

        torch.sqrt(

            1.0 - alpha_hat_t

        )

        *

        noise

    )


    return noisy_images, noise


print()

print("Forward Diffusion Function Created Successfully")


# ============================================================
# SECTION 19 : TEST FORWARD DIFFUSION
# ============================================================

# Select one batch.

test_images = real_images.to(

    device

)


# Create random timesteps for each image.

random_timesteps = torch.randint(

    0,

    TIMESTEPS,

    (

        test_images.shape[0],

    ),

    device=device

)


# Add noise.

noisy_images, actual_noise = forward_diffusion(

    test_images,

    random_timesteps

)


print()

print("=" * 70)

print("FORWARD DIFFUSION TEST")

print("=" * 70)


print()

print("Original Image Shape:", test_images.shape)

print("Noisy Image Shape:", noisy_images.shape)

print("Noise Shape:", actual_noise.shape)


# ============================================================
# SECTION 20 : VISUALIZE IMAGE AT DIFFERENT TIMESTEPS
# ============================================================

# Select one original image.

original_image = X_tensor[0].to(

    device

).unsqueeze(

    0

)


# Different stages of diffusion.

visualization_timesteps = [

    0,

    10,

    25,

    40,

    49

]


fig, axes = plt.subplots(

    1,

    len(visualization_timesteps) + 1,

    figsize=(15, 3)

)


# Display original image.

original_display = original_image[0].cpu().numpy()


original_display = (

    original_display + 1

) / 2


original_display = original_display.reshape(

    IMAGE_HEIGHT,

    IMAGE_WIDTH

)


axes[0].imshow(

    original_display,

    cmap="gray",

    vmin=0,

    vmax=1

)


axes[0].set_title(

    "Original"

)


axes[0].axis(

    "off"

)


# Use the same noise vector so that the
# progressive noising is easier to visualize.

fixed_noise = torch.randn_like(

    original_image

)


for index, timestep in enumerate(

    visualization_timesteps

):


    # Create timestep tensor.

    timestep_tensor = torch.tensor(

        [timestep],

        device=device

    )


    # Get cumulative alpha.

    alpha_hat_t = alpha_hat[

        timestep_tensor

    ].unsqueeze(

        1

    )


    # Create noisy image.

    noisy_image = (

        torch.sqrt(

            alpha_hat_t

        )

        *

        original_image

        +

        torch.sqrt(

            1.0 - alpha_hat_t

        )

        *

        fixed_noise

    )


    # Convert for visualization.

    image = noisy_image[0].cpu().numpy()


    image = (

        image + 1

    ) / 2


    # Keep visualization values valid.

    image = np.clip(

        image,

        0,

        1

    )


    # Reshape to 8x8.

    image = image.reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    # Display.

    axes[index + 1].imshow(

        image,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    axes[index + 1].set_title(

        f"t = {timestep}"

    )


    axes[index + 1].axis(

        "off"

    )


plt.suptitle(

    "Forward Diffusion: Gradual Noise Addition"

)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 21 : DEFINE MODEL CONFIGURATION
# ============================================================

# Latent hidden dimensions for the
# small noise prediction network.

HIDDEN_DIM = 128


print()

print("=" * 70)

print("MODEL CONFIGURATION")

print("=" * 70)


print()

print("Input Dimension:", INPUT_DIM)

print("Hidden Dimension:", HIDDEN_DIM)

print("Batch Size:", BATCH_SIZE)

print("Diffusion Timesteps:", TIMESTEPS)

print("Device:", device)


# ============================================================
# SECTION 22 : FINAL PART-2 SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-2 SUMMARY")

print("=" * 70)


print()

print("Original Data Shape:", X.shape)

print("Diffusion Tensor Shape:", X_tensor.shape)

print("Pixel Range:", f"{X_tensor.min().item()} to {X_tensor.max().item()}")

print("Batch Size:", BATCH_SIZE)

print("Diffusion Steps:", TIMESTEPS)

print("Beta Range:", f"{BETA_START} to {BETA_END}")

print("Hidden Dimension:", HIDDEN_DIM)


# ============================================================
# SECTION 23 : PART-2 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-2 COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("Important Variables Created:")

print()

print("✔ X_normalized")

print("✔ X_diffusion")

print("✔ X_tensor")

print("✔ diffusion_dataset")

print("✔ train_loader")

print("✔ device")

print("✔ TIMESTEPS")

print("✔ betas")

print("✔ alphas")

print("✔ alpha_hat")

print("✔ forward_diffusion")

print("✔ HIDDEN_DIM")


print()

print("READY FOR PART-3")

print("BUILDING THE DIFFUSION NOISE PREDICTION MODEL")
# ============================================================
# PART-3 : BUILDING THE DIFFUSION NOISE PREDICTION MODEL
# PROJECT : AI IMAGE GENERATION USING DIFFUSION MODEL
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import torch

import torch.nn as nn

import torch.optim as optim


print("Part-3 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : DEFINE TIME EMBEDDING DIMENSION
# ============================================================

# The model needs to know at which diffusion
# timestep the image currently exists.
#
# Example:
#
# t = 0   -> Almost clean image
#
# t = 49  -> Highly noisy image

TIME_EMBEDDING_DIM = 32


print()

print("Time Embedding Dimension:", TIME_EMBEDDING_DIM)


# ============================================================
# SECTION 3 : CREATE TIME EMBEDDING FUNCTION
# ============================================================

# This function converts a timestep into a
# normalized numerical representation.
#
# Example:
#
# Timestep 0
#     ↓
# Normalized value = 0.0
#
# Timestep 49
#     ↓
# Normalized value ≈ 1.0
#
# The normalized timestep is then expanded
# into multiple features.


def get_time_embedding(

    timesteps,

    embedding_dim=TIME_EMBEDDING_DIM

):


    # Convert timestep to float.

    timesteps = timesteps.float()


    # Normalize timestep values.

    normalized_time = (

        timesteps

        /

        (TIMESTEPS - 1)

    )


    # Convert shape:
    #
    # (batch_size,)
    #
    # into:
    #
    # (batch_size, 1)

    normalized_time = normalized_time.unsqueeze(

        1

    )


    # Repeat the timestep value to create
    # a simple CPU-friendly embedding.

    time_embedding = normalized_time.repeat(

        1,

        embedding_dim

    )


    return time_embedding


print()

print("Time Embedding Function Created Successfully")


# ============================================================
# SECTION 4 : TEST TIME EMBEDDING
# ============================================================

test_timesteps = torch.tensor(

    [0, 10, 25, 49],

    device=device

)


test_time_embedding = get_time_embedding(

    test_timesteps

)


print()

print("=" * 70)

print("TIME EMBEDDING TEST")

print("=" * 70)


print()

print("Input Timesteps:", test_timesteps)

print("Time Embedding Shape:", test_time_embedding.shape)


# ============================================================
# SECTION 5 : DEFINE DIFFUSION NOISE PREDICTION MODEL
# ============================================================

# The model receives:
#
# 1. Noisy Image
#       Shape: 64
#
# 2. Time Embedding
#       Shape: 32
#
# These are combined together:
#
# 64 + 32 = 96 input features
#
#
# Architecture:
#
# Noisy Image + Time Embedding
#              ↓
#          Linear Layer
#              ↓
#            ReLU
#              ↓
#          Linear Layer
#              ↓
#            ReLU
#              ↓
#        Predicted Noise
#           64 values


class DiffusionModel(

    nn.Module

):


    def __init__(

        self,

        input_dim,

        time_embedding_dim,

        hidden_dim

    ):


        # Initialize parent class.

        super().__init__()


        # Total input features.
        #
        # Image pixels + time embedding.

        total_input_dim = (

            input_dim

            +

            time_embedding_dim

        )


        # Build the neural network.

        self.network = nn.Sequential(


            # First hidden layer.

            nn.Linear(

                total_input_dim,

                hidden_dim

            ),


            # Activation function.

            nn.ReLU(),


            # Second hidden layer.

            nn.Linear(

                hidden_dim,

                hidden_dim

            ),


            # Activation function.

            nn.ReLU(),


            # Output layer.
            #
            # Predict noise for all 64 pixels.

            nn.Linear(

                hidden_dim,

                input_dim

            )

        )


    def forward(

        self,

        noisy_images,

        timesteps

    ):


        # Create time embedding.

        time_embedding = get_time_embedding(

            timesteps,

            TIME_EMBEDDING_DIM

        )


        # Ensure time embedding is on the
        # same device as the noisy images.

        time_embedding = time_embedding.to(

            noisy_images.device

        )


        # Combine noisy image features
        # with timestep information.

        model_input = torch.cat(

            [

                noisy_images,

                time_embedding

            ],

            dim=1

        )


        # Predict the noise.

        predicted_noise = self.network(

            model_input

        )


        return predicted_noise


print()

print("Diffusion Model Class Created Successfully")


# ============================================================
# SECTION 6 : CREATE DIFFUSION MODEL
# ============================================================

model = DiffusionModel(

    input_dim=INPUT_DIM,

    time_embedding_dim=TIME_EMBEDDING_DIM,

    hidden_dim=HIDDEN_DIM

)


# Move the model to the selected device.

model = model.to(

    device

)


print()

print("=" * 70)

print("DIFFUSION MODEL ARCHITECTURE")

print("=" * 70)


print()

print(

    model

)


# ============================================================
# SECTION 7 : COUNT TRAINABLE PARAMETERS
# ============================================================

total_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad

)


print()

print("=" * 70)

print("MODEL PARAMETER INFORMATION")

print("=" * 70)


print()

print(

    "Total Trainable Parameters:",

    total_parameters

)


# ============================================================
# SECTION 8 : DEFINE LOSS FUNCTION
# ============================================================

# The model predicts the actual Gaussian
# noise added to an image.
#
# Therefore, Mean Squared Error is used to
# compare:
#
# Actual Noise
#
# vs
#
# Predicted Noise

loss_function = nn.MSELoss()


print()

print("Loss Function Created Successfully")

print("Loss Function: Mean Squared Error")


# ============================================================
# SECTION 9 : DEFINE OPTIMIZER
# ============================================================

# Adam is used for stable optimization.

LEARNING_RATE = 0.001


optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


print()

print("Optimizer Created Successfully")

print("Optimizer: Adam")

print("Learning Rate:", LEARNING_RATE)


# ============================================================
# SECTION 10 : TEST MODEL INPUT AND OUTPUT
# ============================================================

# Get one batch from the existing DataLoader.

sample_batch = next(

    iter(train_loader)

)


test_images = sample_batch[0].to(

    device

)


# Create random diffusion timesteps.

test_timesteps = torch.randint(

    low=0,

    high=TIMESTEPS,

    size=(

        test_images.shape[0],

    ),

    device=device

)


# Add noise using the forward diffusion
# function created in Part-2.

test_noisy_images, test_actual_noise = forward_diffusion(

    test_images,

    test_timesteps

)


# ============================================================
# SECTION 11 : PREDICT NOISE
# ============================================================

# Pass noisy images and timesteps
# into the model.

test_predicted_noise = model(

    test_noisy_images,

    test_timesteps

)


print()

print("=" * 70)

print("MODEL INPUT AND OUTPUT TEST")

print("=" * 70)


print()

print(

    "Original Image Shape:",

    test_images.shape

)


print(

    "Noisy Image Shape:",

    test_noisy_images.shape

)


print(

    "Actual Noise Shape:",

    test_actual_noise.shape

)


print(

    "Predicted Noise Shape:",

    test_predicted_noise.shape

)


# ============================================================
# SECTION 12 : CALCULATE TEST LOSS
# ============================================================

test_loss = loss_function(

    test_predicted_noise,

    test_actual_noise

)


print()

print("=" * 70)

print("INITIAL MODEL TEST LOSS")

print("=" * 70)


print()

print(

    "Initial MSE Loss:",

    test_loss.item()

)


# ============================================================
# SECTION 13 : TEST ONE TRAINING STEP
# ============================================================

# Put the model into training mode.

model.train()


# Clear old gradients.

optimizer.zero_grad()


# Generate random timesteps.

training_timesteps = torch.randint(

    low=0,

    high=TIMESTEPS,

    size=(

        test_images.shape[0],

    ),

    device=device

)


# Add noise.

training_noisy_images, training_actual_noise = forward_diffusion(

    test_images,

    training_timesteps

)


# Predict noise.

training_predicted_noise = model(

    training_noisy_images,

    training_timesteps

)


# Calculate loss.

training_loss = loss_function(

    training_predicted_noise,

    training_actual_noise

)


# Backpropagation.

training_loss.backward()


# Update model parameters.

optimizer.step()


print()

print("=" * 70)

print("ONE TRAINING STEP COMPLETED")

print("=" * 70)


print()

print(

    "Training Loss:",

    training_loss.item()

)


# ============================================================
# SECTION 14 : TEST MODEL AFTER ONE UPDATE
# ============================================================

model.eval()


with torch.no_grad():


    updated_predicted_noise = model(

        test_noisy_images,

        test_timesteps

    )


updated_test_loss = loss_function(

    updated_predicted_noise,

    test_actual_noise

)


print()

print("=" * 70)

print("MODEL TEST AFTER ONE UPDATE")

print("=" * 70)


print()

print(

    "Updated Test Loss:",

    updated_test_loss.item()

)


# ============================================================
# SECTION 15 : DEFINE FINAL MODEL CONFIGURATION
# ============================================================

print()

print("=" * 70)

print("FINAL MODEL CONFIGURATION")

print("=" * 70)


print()

print(

    "Input Dimension:",

    INPUT_DIM

)


print(

    "Time Embedding Dimension:",

    TIME_EMBEDDING_DIM

)


print(

    "Hidden Dimension:",

    HIDDEN_DIM

)


print(

    "Diffusion Timesteps:",

    TIMESTEPS

)


print(

    "Learning Rate:",

    LEARNING_RATE

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
# SECTION 16 : PART-3 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-3 COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("Completed:")

print()

print("✔ Time Embedding")

print("✔ Diffusion Noise Prediction Model")

print("✔ Model Architecture")

print("✔ Parameter Analysis")

print("✔ MSE Loss Function")

print("✔ Adam Optimizer")

print("✔ Forward Diffusion Test")

print("✔ Noise Prediction Test")

print("✔ One Training Step Test")


print()

print("READY FOR PART-4")

print("MODEL TRAINING, REVERSE DIFFUSION & AI IMAGE GENERATION")
# ============================================================
# PART-4 : MODEL TRAINING, REVERSE DIFFUSION
#          & AI IMAGE GENERATION
# PROJECT : AI IMAGE GENERATION USING DIFFUSION MODEL
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
# SECTION 2 : SET TRAINING CONFIGURATION
# ============================================================

# Keep epochs limited for CPU-friendly training.

EPOCHS = 50


print()

print("=" * 70)

print("TRAINING CONFIGURATION")

print("=" * 70)

print()

print("Epochs:", EPOCHS)

print("Batch Size:", BATCH_SIZE)

print("Learning Rate:", LEARNING_RATE)

print("Diffusion Timesteps:", TIMESTEPS)

print("Device:", device)


# ============================================================
# SECTION 3 : CREATE TRAINING HISTORY
# ============================================================

training_losses = []


print()

print("Training History Created Successfully")


# ============================================================
# SECTION 4 : START DIFFUSION MODEL TRAINING
# ============================================================

print()

print("=" * 70)

print("STARTING DIFFUSION MODEL TRAINING")

print("=" * 70)


for epoch in range(EPOCHS):


    # Put model into training mode.

    model.train()


    # Track total loss for the epoch.

    total_loss = 0.0


    # Count total batches.

    total_batches = 0


    # --------------------------------------------------------
    # LOOP THROUGH TRAINING BATCHES
    # --------------------------------------------------------

    for batch in train_loader:


        # Extract image data.

        images = batch[0].to(

            device

        )


        # Get current batch size.

        current_batch_size = images.shape[0]


        # ----------------------------------------------------
        # CREATE RANDOM TIMESTEPS
        # ----------------------------------------------------

        # Each image receives a random
        # diffusion timestep.

        timesteps = torch.randint(

            low=0,

            high=TIMESTEPS,

            size=(current_batch_size,),

            device=device

        )


        # ----------------------------------------------------
        # FORWARD DIFFUSION
        # ----------------------------------------------------

        # Add Gaussian noise to the images.

        noisy_images, actual_noise = forward_diffusion(

            images,

            timesteps

        )


        # ----------------------------------------------------
        # CLEAR OLD GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # PREDICT THE NOISE
        # ----------------------------------------------------

        predicted_noise = model(

            noisy_images,

            timesteps

        )


        # ----------------------------------------------------
        # CALCULATE LOSS
        # ----------------------------------------------------

        # Compare predicted noise with
        # the actual noise.

        loss = loss_function(

            predicted_noise,

            actual_noise

        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # Update model parameters.

        optimizer.step()


        # ----------------------------------------------------
        # STORE LOSS
        # ----------------------------------------------------

        total_loss += loss.item()

        total_batches += 1


    # ========================================================
    # CALCULATE AVERAGE EPOCH LOSS
    # ========================================================

    average_loss = (

        total_loss

        /

        total_batches

    )


    # Store epoch loss.

    training_losses.append(

        average_loss

    )


    # ========================================================
    # DISPLAY TRAINING PROGRESS
    # ========================================================

    if (

        epoch == 0

        or

        (epoch + 1) % 5 == 0

    ):


        print(

            f"Epoch [{epoch + 1}/{EPOCHS}] "

            f"| Loss: {average_loss:.6f}"

        )


print()

print("=" * 70)

print("DIFFUSION MODEL TRAINING COMPLETED")

print("=" * 70)


# ============================================================
# SECTION 5 : VISUALIZE TRAINING LOSS
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.plot(

    training_losses

)


plt.title(

    "Diffusion Model Training Loss"

)


plt.xlabel(

    "Epoch"

)


plt.ylabel(

    "MSE Loss"

)


plt.grid(

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 6 : DISPLAY LOSS INFORMATION
# ============================================================

print()

print("=" * 70)

print("TRAINING LOSS SUMMARY")

print("=" * 70)


print()

print(

    "Initial Training Loss:",

    round(

        training_losses[0],

        6

    )

)


print(

    "Final Training Loss:",

    round(

        training_losses[-1],

        6

    )

)


# ============================================================
# SECTION 7 : CREATE REVERSE DIFFUSION FUNCTION
# ============================================================

# Reverse diffusion starts with random noise
# and gradually removes predicted noise.


def reverse_diffusion_step(

    current_image,

    timestep

):


    # Create a timestep tensor for
    # every image in the batch.

    timestep_tensor = torch.full(

        (

            current_image.shape[0],

        ),

        timestep,

        device=device,

        dtype=torch.long

    )


    # Predict the noise present
    # in the current image.

    predicted_noise = model(

        current_image,

        timestep_tensor

    )


    # Get diffusion values.

    alpha_t = alphas[

        timestep

    ]


    alpha_hat_t = alpha_hat[

        timestep

    ]


    beta_t = betas[

        timestep

    ]


    # Calculate the mean for the
    # reverse diffusion process.

    mean = (

        1

        /

        torch.sqrt(

            alpha_t

        )

    ) * (

        current_image

        -

        (

            beta_t

            /

            torch.sqrt(

                1.0 - alpha_hat_t

            )

        )

        *

        predicted_noise

    )


    # --------------------------------------------------------
    # ADD SMALL RANDOM NOISE
    # --------------------------------------------------------

    # For the final timestep,
    # no additional noise is needed.

    if timestep > 0:


        random_noise = torch.randn_like(

            current_image

        )


        current_image = (

            mean

            +

            torch.sqrt(

                beta_t

            )

            *

            random_noise

        )


    else:


        current_image = mean


    return current_image


print()

print("Reverse Diffusion Function Created Successfully")


# ============================================================
# SECTION 8 : TEST REVERSE DIFFUSION
# ============================================================

model.eval()


# Start with one random noisy image.

test_noise = torch.randn(

    1,

    INPUT_DIM,

    device=device

)


with torch.no_grad():


    test_result = reverse_diffusion_step(

        test_noise,

        TIMESTEPS - 1

    )


print()

print("=" * 70)

print("REVERSE DIFFUSION TEST")

print("=" * 70)


print()

print("Input Noise Shape:", test_noise.shape)

print("Output Shape:", test_result.shape)


# ============================================================
# SECTION 9 : CREATE COMPLETE IMAGE GENERATION FUNCTION
# ============================================================

def generate_images(

    number_of_images=10,

    save_steps=False

):


    # Put model into evaluation mode.

    model.eval()


    # Start with completely random Gaussian noise.

    generated_images = torch.randn(

        number_of_images,

        INPUT_DIM,

        device=device

    )


    # Store intermediate images if needed.

    generation_steps = []


    # Disable gradient calculation.

    with torch.no_grad():


        # Start from the noisiest timestep
        # and move backward to timestep 0.

        for timestep in reversed(

            range(TIMESTEPS)

        ):


            generated_images = reverse_diffusion_step(

                generated_images,

                timestep

            )


            # Save selected steps for visualization.

            if save_steps:


                if timestep in [

                    49,

                    40,

                    30,

                    20,

                    10,

                    0

                ]:


                    generation_steps.append(

                        (

                            timestep,

                            generated_images.clone().cpu()

                        )

                    )


    if save_steps:


        return generated_images.cpu(), generation_steps


    return generated_images.cpu()


print()

print("Complete Image Generation Function Created Successfully")


# ============================================================
# SECTION 10 : GENERATE FINAL SYNTHETIC DIGITS
# ============================================================

NUM_GENERATED_IMAGES = 20


print()

print("Generating Synthetic Images...")


generated_images = generate_images(

    number_of_images=NUM_GENERATED_IMAGES

)


print()

print("Synthetic Images Generated Successfully")


print(

    "Generated Image Tensor Shape:",

    generated_images.shape

)


# ============================================================
# SECTION 11 : VISUALIZE FINAL GENERATED IMAGES
# ============================================================

fig, axes = plt.subplots(

    4,

    5,

    figsize=(10, 8)

)


for index in range(NUM_GENERATED_IMAGES):


    # Get one generated image.

    image = generated_images[

        index

    ].numpy()


    # Convert values from approximately
    # -1,+1 back to 0,1.

    image = (

        image + 1.0

    ) / 2.0


    # Keep pixel values valid.

    image = np.clip(

        image,

        0,

        1

    )


    # Reshape into 8 x 8 image.

    image = image.reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    # Select subplot.

    ax = axes.flat[

        index

    ]


    # Display image.

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

    "Final AI Generated Images Using Diffusion Model"

)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 12 : VISUALIZE THE DENOISING PROCESS
# ============================================================

print()

print("Generating Denoising Visualization...")


# Generate one image and save
# intermediate reverse diffusion steps.

final_image, denoising_steps = generate_images(

    number_of_images=1,

    save_steps=True

)


# Number of saved stages.

number_of_steps = len(

    denoising_steps

)


fig, axes = plt.subplots(

    1,

    number_of_steps,

    figsize=(15, 3)

)


for index, (

    timestep,

    image_tensor

) in enumerate(

    denoising_steps

):


    # Get generated image.

    image = image_tensor[

        0

    ].numpy()


    # Convert from -1,+1 to 0,1.

    image = (

        image + 1.0

    ) / 2.0


    # Clip invalid visualization values.

    image = np.clip(

        image,

        0,

        1

    )


    # Reshape to 8 x 8.

    image = image.reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    # Display.

    axes[index].imshow(

        image,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    axes[index].set_title(

        f"t = {timestep}"

    )


    axes[index].axis(

        "off"

    )


plt.suptitle(

    "Reverse Diffusion: Noise to Generated Image"

)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 13 : GENERATE NEW SAMPLE USING REUSABLE FUNCTION
# ============================================================

new_generated_images = generate_images(

    number_of_images=10

)


print()

print(

    "Reusable Generation Function Tested Successfully"

)


print(

    "New Generated Images Shape:",

    new_generated_images.shape

)


# ============================================================
# SECTION 14 : VISUALIZE NEW GENERATED SAMPLES
# ============================================================

fig, axes = plt.subplots(

    2,

    5,

    figsize=(10, 5)

)


for index in range(10):


    image = new_generated_images[

        index

    ].numpy()


    image = (

        image + 1.0

    ) / 2.0


    image = np.clip(

        image,

        0,

        1

    )


    image = image.reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    ax = axes.flat[

        index

    ]


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

    "New AI Generated Digit Samples"

)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 15 : SAVE THE TRAINED DIFFUSION MODEL
# ============================================================

MODEL_SAVE_PATH = "diffusion_digits_model.pth"


torch.save(

    {

        "model_state_dict":

            model.state_dict(),


        "optimizer_state_dict":

            optimizer.state_dict(),


        "input_dim":

            INPUT_DIM,


        "hidden_dim":

            HIDDEN_DIM,


        "time_embedding_dim":

            TIME_EMBEDDING_DIM,


        "timesteps":

            TIMESTEPS,


        "training_losses":

            training_losses

    },

    MODEL_SAVE_PATH

)


print()

print("=" * 70)

print("MODEL SAVED SUCCESSFULLY")

print("=" * 70)


print()

print(

    "Model Saved As:",

    MODEL_SAVE_PATH

)


# ============================================================
# SECTION 16 : CREATE FINAL PROJECT SUMMARY
# ============================================================

final_project_info = {


    "Project":

        "AI Image Generation using Diffusion Model",


    "Dataset":

        "Scikit-learn Digits Dataset",


    "Total Images":

        X.shape[0],


    "Image Size":

        f"{IMAGE_HEIGHT} x {IMAGE_WIDTH}",


    "Input Dimension":

        INPUT_DIM,


    "Diffusion Timesteps":

        TIMESTEPS,


    "Hidden Dimension":

        HIDDEN_DIM,


    "Batch Size":

        BATCH_SIZE,


    "Epochs":

        EPOCHS,


    "Learning Rate":

        LEARNING_RATE,


    "Initial Loss":

        round(

            training_losses[0],

            6

        ),


    "Final Loss":

        round(

            training_losses[-1],

            6

        ),


    "Saved Model":

        MODEL_SAVE_PATH

}


print()

print("=" * 70)

print("FINAL PROJECT SUMMARY")

print("=" * 70)


print()


for key, value in final_project_info.items():


    print(

        f"{key}: {value}"

    )


# ============================================================
# SECTION 17 : PROJECT COMPLETION
# ============================================================

print()

print("=" * 70)

print("DAY 46/100 PROJECT COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("PROJECT: AI IMAGE GENERATION USING DIFFUSION MODEL")


print()

print("COMPLETE PIPELINE:")


print()

print("✔ Built-in Scikit-learn Digits Dataset")

print("✔ Dataset Exploration")

print("✔ Pixel Normalization")

print("✔ Diffusion Noise Schedule")

print("✔ Forward Diffusion")

print("✔ Gaussian Noise Addition")

print("✔ Time Embedding")

print("✔ Noise Prediction Neural Network")

print("✔ MSE Loss")

print("✔ Model Training")

print("✔ Training Loss Visualization")

print("✔ Reverse Diffusion")

print("✔ Noise-to-Image Generation")

print("✔ AI Generated Digit Visualization")

print("✔ Denoising Process Visualization")

print("✔ Reusable Image Generation Function")

print("✔ Trained Model Saving")


print()

print("END-TO-END AI IMAGE GENERATION PROJECT COMPLETE 🚀")
