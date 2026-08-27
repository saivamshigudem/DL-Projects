# ============================================================
# PART-1 : DATASET LOADING & EXPLORATORY DATA ANALYSIS
# PROJECT : IMAGE GENERATION USING GANs
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

# The Digits dataset contains handwritten digit images.
#
# Total Images : 1797
#
# Image Size   : 8 x 8
#
# Classes      : Digits 0 to 9
#
# Pixel Range  : 0 to 16

digits = load_digits()


print()

print("Digits Dataset Loaded Successfully")


# ============================================================
# SECTION 3 : EXTRACT IMAGE DATA AND LABELS
# ============================================================

# X contains flattened image pixel values.
#
# Shape:
#
# (1797, 64)
#
# Each image has:
#
# 8 x 8 = 64 pixels

X = digits.data


# y contains the actual digit labels.

y = digits.target


# images contains the original 8 x 8 image format.

images = digits.images


print()

print("Image Data and Labels Extracted Successfully")


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

print("Classes:", np.unique(y))


# ============================================================
# SECTION 5 : CREATE DATAFRAME
# ============================================================

# Convert the flattened pixel data into a DataFrame.
#
# Each column represents one pixel.

df = pd.DataFrame(

    X,

    columns=[

        f"pixel_{index}"

        for index in range(X.shape[1])

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


display(df.head())


# ============================================================
# SECTION 7 : CHECK DATA TYPES
# ============================================================

print()

print("=" * 70)

print("DATA TYPES")

print("=" * 70)


print()

print(df.dtypes)


# ============================================================
# SECTION 8 : CHECK MISSING VALUES
# ============================================================

print()

print("=" * 70)

print("MISSING VALUE ANALYSIS")

print("=" * 70)


missing_values = df.isnull().sum()


print()

print(missing_values)


print()

print("Total Missing Values:", missing_values.sum())


# ============================================================
# SECTION 9 : STATISTICAL SUMMARY
# ============================================================

print()

print("=" * 70)

print("STATISTICAL SUMMARY")

print("=" * 70)


display(df.describe())


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

print(class_distribution)


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


    # Find the first occurrence of each digit.

    index = np.where(

        y == digit

    )[0][0]


    # Get the corresponding image.

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


    # Display digit label.

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

print("PIXEL VALUE ANALYSIS")

print("=" * 70)


print()

print("Minimum Pixel Value:", X.min())

print("Maximum Pixel Value:", X.max())


print()

print(

    "Pixel values currently range from",

    X.min(),

    "to",

    X.max()

)


print()

print(

    "The pixel values will be normalized in Part-2."

)


# ============================================================
# SECTION 15 : CHECK IMAGE DIMENSIONS
# ============================================================

IMAGE_HEIGHT = images.shape[1]

IMAGE_WIDTH = images.shape[2]

INPUT_DIM = X.shape[1]


print()

print("=" * 70)

print("IMAGE DIMENSION INFORMATION")

print("=" * 70)


print()

print("Image Height:", IMAGE_HEIGHT)

print("Image Width:", IMAGE_WIDTH)

print("Flattened Input Dimension:", INPUT_DIM)


# ============================================================
# SECTION 16 : DISPLAY RAW PIXEL MATRIX
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
# SECTION 17 : DISPLAY PIXEL INTENSITY HISTOGRAM
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

print("Input Dimension:", INPUT_DIM)

print("Number of Classes:", len(np.unique(y)))

print("Classes:", np.unique(y))

print("Missing Values:", missing_values.sum())

print("Pixel Value Range:", f"{X.min()} to {X.max()}")


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

print("DATA PREPROCESSING & GAN DATALOADER PREPARATION")
# ============================================================
# PART-2 : DATA PREPROCESSING & GAN DATALOADER PREPARATION
# PROJECT : IMAGE GENERATION USING GANs
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np

import torch

from torch.utils.data import TensorDataset, DataLoader


print("Part-2 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : SET RANDOM SEED
# ============================================================

# Setting a random seed helps us get more
# consistent results when running the project.

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
# SECTION 3 : CHECK ORIGINAL PIXEL VALUE RANGE
# ============================================================

print()

print("=" * 70)

print("ORIGINAL PIXEL VALUE RANGE")

print("=" * 70)


print()

print("Minimum Pixel Value:", X.min())

print("Maximum Pixel Value:", X.max())


# ============================================================
# SECTION 4 : NORMALIZE PIXELS FROM 0-16 TO 0-1
# ============================================================

# The sklearn Digits dataset has pixel values
# between 0 and 16.
#
# Dividing by 16 converts them to:
#
# 0 to 1

X_normalized = X.astype(
    np.float32
) / 16.0


print()

print("Pixel Normalization from 0-16 to 0-1 Completed")


# ============================================================
# SECTION 5 : NORMALIZE PIXELS FROM 0-1 TO -1 TO +1
# ============================================================

# GAN Generator will use Tanh activation.
#
# Tanh produces output values between:
#
# -1 and +1
#
# Therefore, real training images should also
# be converted into the same range.
#
# Formula:
#
# X_gan = (X_normalized * 2) - 1

X_gan = (
    X_normalized * 2.0
) - 1.0


print()

print("GAN Normalization from 0-1 to -1 to +1 Completed")


# ============================================================
# SECTION 6 : CHECK FINAL PIXEL VALUE RANGE
# ============================================================

print()

print("=" * 70)

print("FINAL GAN PIXEL VALUE RANGE")

print("=" * 70)


print()

print("Minimum Pixel Value:", X_gan.min())

print("Maximum Pixel Value:", X_gan.max())


# ============================================================
# SECTION 7 : CONVERT NUMPY DATA TO PYTORCH TENSOR
# ============================================================

# Convert the normalized image data
# into a PyTorch tensor.

X_tensor = torch.tensor(
    X_gan,
    dtype=torch.float32
)


print()

print("PyTorch Tensor Created Successfully")


# ============================================================
# SECTION 8 : CHECK TENSOR INFORMATION
# ============================================================

print()

print("=" * 70)

print("PYTORCH TENSOR INFORMATION")

print("=" * 70)


print()

print("Tensor Shape:", X_tensor.shape)

print("Tensor Data Type:", X_tensor.dtype)

print("Minimum Value:", X_tensor.min().item())

print("Maximum Value:", X_tensor.max().item())


# ============================================================
# SECTION 9 : CREATE PYTORCH DATASET
# ============================================================

# A GAN does not require labels for training.
#
# The discriminator only needs real images.
#
# Therefore, we create a dataset using
# only the image tensor.

gan_dataset = TensorDataset(
    X_tensor
)


print()

print("GAN Dataset Created Successfully")


# ============================================================
# SECTION 10 : SET CPU-FRIENDLY BATCH SIZE
# ============================================================

# A small batch size is suitable for
# a slow CPU and low memory.

BATCH_SIZE = 32


print()

print("Batch Size:", BATCH_SIZE)


# ============================================================
# SECTION 11 : CREATE GAN DATALOADER
# ============================================================

train_loader = DataLoader(

    gan_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


print()

print("GAN DataLoader Created Successfully")


# ============================================================
# SECTION 12 : CHECK DATALOADER INFORMATION
# ============================================================

print()

print("=" * 70)

print("DATALOADER INFORMATION")

print("=" * 70)


print()

print("Total Images:", len(gan_dataset))

print("Total Batches:", len(train_loader))

print("Batch Size:", BATCH_SIZE)


# ============================================================
# SECTION 13 : INSPECT ONE BATCH
# ============================================================

sample_batch = next(
    iter(train_loader)
)


# TensorDataset returns data as a tuple.
# Therefore, we take the first element.

sample_images = sample_batch[0]


print()

print("=" * 70)

print("SAMPLE BATCH INFORMATION")

print("=" * 70)


print()

print("Batch Shape:", sample_images.shape)

print("Batch Minimum Value:", sample_images.min().item())

print("Batch Maximum Value:", sample_images.max().item())


# ============================================================
# SECTION 14 : CHECK COMPUTATION DEVICE
# ============================================================

# The code automatically checks for a GPU.
#
# If no GPU is available, CPU will be used.

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
# SECTION 15 : VISUALIZE NORMALIZED GAN IMAGES
# ============================================================

# GAN images are currently in the range:
#
# -1 to +1
#
# To visualize them, convert them back
# to the range 0 to 1.
#
# Formula:
#
# image = (image + 1) / 2

fig, axes = plt.subplots(

    2,

    5,

    figsize=(10, 5)

)


for index in range(10):


    # Get one normalized image.

    image = X_tensor[
        index
    ].numpy()


    # Convert from -1,+1 to 0,1.

    image = (
        image + 1
    ) / 2


    # Reshape from 64 pixels to 8x8.

    image = image.reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )


    # Display the image.

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
        f"Digit: {y[index]}"
    )


    ax.axis(
        "off"
    )


plt.tight_layout()

plt.show()


# ============================================================
# SECTION 16 : DEFINE GAN CONFIGURATION
# ============================================================

# Latent dimension represents the size
# of the random noise vector that will
# be given to the Generator.

LATENT_DIM = 32


print()

print("=" * 70)

print("GAN CONFIGURATION")

print("=" * 70)


print()

print("Input Dimension:", INPUT_DIM)

print("Image Height:", IMAGE_HEIGHT)

print("Image Width:", IMAGE_WIDTH)

print("Latent Dimension:", LATENT_DIM)

print("Batch Size:", BATCH_SIZE)

print("Device:", device)


# ============================================================
# SECTION 17 : FINAL DATA PREPARATION SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-2 DATA PREPARATION SUMMARY")

print("=" * 70)


print()

print("Dataset:", "sklearn Digits Dataset")

print("Total Images:", len(X_tensor))

print("Image Shape:", f"{IMAGE_HEIGHT} x {IMAGE_WIDTH}")

print("Input Dimension:", INPUT_DIM)

print("Original Pixel Range:", f"{X.min()} to {X.max()}")

print("GAN Pixel Range:", f"{X_gan.min()} to {X_gan.max()}")

print("Batch Size:", BATCH_SIZE)

print("Latent Dimension:", LATENT_DIM)


# ============================================================
# SECTION 18 : PART-2 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-2 COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("Important Variables Created:")

print()

print("✔ X_normalized")

print("✔ X_gan")

print("✔ X_tensor")

print("✔ gan_dataset")

print("✔ train_loader")

print("✔ BATCH_SIZE")

print("✔ LATENT_DIM")

print("✔ device")


print()

print("READY FOR PART-3")

print("BUILDING THE GENERATOR AND DISCRIMINATOR NETWORKS")
# ============================================================
# PART-3 : BUILDING THE GENERATOR & DISCRIMINATOR
# PROJECT : IMAGE GENERATION USING GANs
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
# SECTION 2 : DEFINE GENERATOR NETWORK
# ============================================================

# The Generator receives a random noise vector
# and converts it into a fake handwritten digit.
#
# Architecture:
#
# Random Noise
#      ↓
# Latent Dimension = 32
#      ↓
# Hidden Layer = 64
#      ↓
# Hidden Layer = 128
#      ↓
# Output Layer = 64 Pixels
#      ↓
# Tanh Activation
#      ↓
# Generated 8 x 8 Image


class Generator(nn.Module):


    def __init__(
        self,
        latent_dim,
        output_dim
    ):


        # Initialize the parent PyTorch class.

        super().__init__()


        # Create the Generator network.

        self.network = nn.Sequential(


            # First hidden layer.

            nn.Linear(

                latent_dim,

                64

            ),


            # ReLU activation.

            nn.ReLU(),


            # Second hidden layer.

            nn.Linear(

                64,

                128

            ),


            # ReLU activation.

            nn.ReLU(),


            # Output layer.
            #
            # Generates 64 pixel values.

            nn.Linear(

                128,

                output_dim

            ),


            # Tanh produces values between:
            #
            # -1 and +1
            #
            # This matches the normalized
            # training images from Part-2.

            nn.Tanh()

        )


    def forward(
        self,
        noise
    ):


        # Pass random noise through
        # the Generator.

        generated_image = self.network(

            noise

        )


        return generated_image


print()

print("Generator Class Created Successfully")


# ============================================================
# SECTION 3 : DEFINE DISCRIMINATOR NETWORK
# ============================================================

# The Discriminator receives an image
# and predicts whether it is:
#
# 1 → Real
#
# 0 → Fake
#
# Architecture:
#
# Input Image
# 64 Pixels
#      ↓
# Hidden Layer = 128
#      ↓
# Hidden Layer = 64
#      ↓
# Output = 1 Value


class Discriminator(nn.Module):


    def __init__(
        self,
        input_dim
    ):


        # Initialize the parent PyTorch class.

        super().__init__()


        # Create the Discriminator network.

        self.network = nn.Sequential(


            # First hidden layer.

            nn.Linear(

                input_dim,

                128

            ),


            # LeakyReLU helps GAN training
            # and avoids completely inactive neurons.

            nn.LeakyReLU(

                0.2

            ),


            # Second hidden layer.

            nn.Linear(

                128,

                64

            ),


            # LeakyReLU activation.

            nn.LeakyReLU(

                0.2

            ),


            # Final output layer.

            nn.Linear(

                64,

                1

            )

        )


    def forward(
        self,
        image
    ):


        # Pass image through the
        # Discriminator.

        output = self.network(

            image

        )


        return output


print()

print("Discriminator Class Created Successfully")


# ============================================================
# SECTION 4 : CREATE GENERATOR MODEL
# ============================================================

generator = Generator(

    latent_dim=LATENT_DIM,

    output_dim=INPUT_DIM

)


# Move Generator to the selected device.

generator = generator.to(

    device

)


print()

print("=" * 70)

print("GENERATOR MODEL")

print("=" * 70)


print()

print(generator)


# ============================================================
# SECTION 5 : CREATE DISCRIMINATOR MODEL
# ============================================================

discriminator = Discriminator(

    input_dim=INPUT_DIM

)


# Move Discriminator to the selected device.

discriminator = discriminator.to(

    device

)


print()

print("=" * 70)

print("DISCRIMINATOR MODEL")

print("=" * 70)


print()

print(discriminator)


# ============================================================
# SECTION 6 : COUNT GENERATOR PARAMETERS
# ============================================================

generator_parameters = sum(

    parameter.numel()

    for parameter in generator.parameters()

    if parameter.requires_grad

)


print()

print("=" * 70)

print("GENERATOR PARAMETERS")

print("=" * 70)


print()

print(

    "Total Trainable Parameters:",

    generator_parameters

)


# ============================================================
# SECTION 7 : COUNT DISCRIMINATOR PARAMETERS
# ============================================================

discriminator_parameters = sum(

    parameter.numel()

    for parameter in discriminator.parameters()

    if parameter.requires_grad

)


print()

print("=" * 70)

print("DISCRIMINATOR PARAMETERS")

print("=" * 70)


print()

print(

    "Total Trainable Parameters:",

    discriminator_parameters

)


# ============================================================
# SECTION 8 : DEFINE GAN LOSS FUNCTION
# ============================================================

# BCEWithLogitsLoss combines:
#
# Sigmoid Activation
#
# +
#
# Binary Cross Entropy Loss
#
# We use it instead of adding Sigmoid
# inside the Discriminator.

adversarial_loss = nn.BCEWithLogitsLoss()


print()

print("GAN Adversarial Loss Created Successfully")

print("Loss Function: BCEWithLogitsLoss")


# ============================================================
# SECTION 9 : SET CPU-FRIENDLY LEARNING RATES
# ============================================================

# Both Generator and Discriminator
# will use separate Adam optimizers.

LEARNING_RATE = 0.0002


# ============================================================
# SECTION 10 : CREATE GENERATOR OPTIMIZER
# ============================================================

generator_optimizer = optim.Adam(

    generator.parameters(),

    lr=LEARNING_RATE,

    betas=(0.5, 0.999)

)


print()

print("Generator Optimizer Created Successfully")


# ============================================================
# SECTION 11 : CREATE DISCRIMINATOR OPTIMIZER
# ============================================================

discriminator_optimizer = optim.Adam(

    discriminator.parameters(),

    lr=LEARNING_RATE,

    betas=(0.5, 0.999)

)


print()

print("Discriminator Optimizer Created Successfully")


# ============================================================
# SECTION 12 : CREATE RANDOM NOISE
# ============================================================

# Random noise acts as the input
# to the Generator.
#
# Shape:
#
# Number of Samples x LATENT_DIM

TEST_BATCH_SIZE = 5


test_noise = torch.randn(

    TEST_BATCH_SIZE,

    LATENT_DIM

).to(

    device

)


print()

print("=" * 70)

print("RANDOM NOISE INFORMATION")

print("=" * 70)


print()

print(

    "Noise Shape:",

    test_noise.shape

)


# ============================================================
# SECTION 13 : TEST THE GENERATOR
# ============================================================

# Generate fake images from
# random noise.

generator.eval()


with torch.no_grad():


    test_generated_images = generator(

        test_noise

    )


print()

print("=" * 70)

print("GENERATOR OUTPUT TEST")

print("=" * 70)


print()

print(

    "Generated Image Shape:",

    test_generated_images.shape

)


print(

    "Minimum Generated Pixel Value:",

    test_generated_images.min().item()

)


print(

    "Maximum Generated Pixel Value:",

    test_generated_images.max().item()

)


# ============================================================
# SECTION 14 : VISUALIZE INITIAL GENERATOR OUTPUT
# ============================================================

# At this stage, the Generator is not trained.
#
# Therefore, generated images will look
# like random patterns.
#
# After training in Part-4, the images
# should start looking more like digits.


import matplotlib.pyplot as plt


fig, axes = plt.subplots(

    1,

    TEST_BATCH_SIZE,

    figsize=(10, 3)

)


for index in range(TEST_BATCH_SIZE):


    # Get one generated image.

    image = test_generated_images[

        index

    ].cpu().numpy()


    # Convert from -1,+1 to 0,1.

    image = (

        image + 1

    ) / 2


    # Reshape 64 pixels into 8 x 8.

    image = image.reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    # Display image.

    axes[index].imshow(

        image,

        cmap="gray",

        vmin=0,

        vmax=1

    )


    axes[index].set_title(

        f"Fake {index + 1}"

    )


    axes[index].axis(

        "off"

    )


plt.suptitle(

    "Generator Output Before Training"

)


plt.tight_layout()

plt.show()


# ============================================================
# SECTION 15 : TEST THE DISCRIMINATOR WITH REAL IMAGES
# ============================================================

# Get one real batch from the DataLoader.

sample_batch = next(

    iter(train_loader)

)


real_images = sample_batch[0].to(

    device

)


# Set Discriminator to evaluation mode.

discriminator.eval()


with torch.no_grad():


    real_predictions = discriminator(

        real_images

    )


print()

print("=" * 70)

print("DISCRIMINATOR REAL IMAGE TEST")

print("=" * 70)


print()

print(

    "Real Image Input Shape:",

    real_images.shape

)


print(

    "Discriminator Output Shape:",

    real_predictions.shape

)


print()

print(

    "First 5 Raw Predictions:"

)


print(

    real_predictions[:5]

)


# ============================================================
# SECTION 16 : TEST DISCRIMINATOR WITH FAKE IMAGES
# ============================================================

with torch.no_grad():


    fake_predictions = discriminator(

        test_generated_images

    )


print()

print("=" * 70)

print("DISCRIMINATOR FAKE IMAGE TEST")

print("=" * 70)


print()

print(

    "Fake Image Input Shape:",

    test_generated_images.shape

)


print(

    "Discriminator Output Shape:",

    fake_predictions.shape

)


print()

print(

    "Fake Image Raw Predictions:"

)


print(

    fake_predictions

)


# ============================================================
# SECTION 17 : CHECK PROBABILITIES
# ============================================================

# BCEWithLogitsLoss expects raw logits.
#
# For visualization only, we can convert
# logits into probabilities using Sigmoid.

real_probabilities = torch.sigmoid(

    real_predictions

)


fake_probabilities = torch.sigmoid(

    fake_predictions

)


print()

print("=" * 70)

print("DISCRIMINATOR PROBABILITY TEST")

print("=" * 70)


print()

print(

    "Real Image Probabilities:"
)


print(

    real_probabilities[:5]

)


print()

print(

    "Fake Image Probabilities:"
)


print(

    fake_probabilities

)


# ============================================================
# SECTION 18 : DEFINE GAN CONFIGURATION SUMMARY
# ============================================================

print()

print("=" * 70)

print("GAN MODEL CONFIGURATION")

print("=" * 70)


print()

print(

    "Generator Input Dimension:",

    LATENT_DIM

)


print(

    "Generator Output Dimension:",

    INPUT_DIM

)


print(

    "Discriminator Input Dimension:",

    INPUT_DIM

)


print(

    "Discriminator Output Dimension:",

    1

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
# SECTION 19 : PART-3 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-3 COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("Completed:")

print()

print("✔ Generator Architecture")

print("✔ Discriminator Architecture")

print("✔ Model Initialization")

print("✔ Parameter Analysis")

print("✔ Random Noise Generation")

print("✔ Fake Image Generation")

print("✔ Real Image Testing")

print("✔ Fake Image Testing")

print("✔ BCEWithLogitsLoss")

print("✔ Generator Optimizer")

print("✔ Discriminator Optimizer")


print()

print("READY FOR PART-4")

print("GAN TRAINING, SYNTHETIC IMAGE GENERATION & FINAL EVALUATION")
# ============================================================
# PART-4 : GAN TRAINING, IMAGE GENERATION & FINAL EVALUATION
# PROJECT : IMAGE GENERATION USING GANs
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

# Keeping epochs small because the project
# is designed for a slow CPU.

EPOCHS = 50


# We will save generated images at selected epochs.

DISPLAY_EPOCHS = [

    1,

    10,

    25,

    50

]


print()

print("=" * 70)

print("GAN TRAINING CONFIGURATION")

print("=" * 70)


print()

print("Epochs:", EPOCHS)

print("Batch Size:", BATCH_SIZE)

print("Latent Dimension:", LATENT_DIM)

print("Learning Rate:", LEARNING_RATE)

print("Device:", device)


# ============================================================
# SECTION 3 : CREATE FIXED NOISE
# ============================================================

# Fixed noise is used to check how the same
# generated images improve during training.

NUM_FIXED_SAMPLES = 10


fixed_noise = torch.randn(

    NUM_FIXED_SAMPLES,

    LATENT_DIM

).to(

    device

)


print()

print("Fixed Noise Created Successfully")

print("Fixed Noise Shape:", fixed_noise.shape)


# ============================================================
# SECTION 4 : CREATE TRAINING HISTORY
# ============================================================

generator_losses = []

discriminator_losses = []


print()

print("Training History Lists Created Successfully")


# ============================================================
# SECTION 5 : START GAN TRAINING
# ============================================================

print()

print("=" * 70)

print("STARTING GAN TRAINING")

print("=" * 70)


for epoch in range(EPOCHS):


    # Put both models into training mode.

    generator.train()

    discriminator.train()


    # Track total losses for this epoch.

    total_generator_loss = 0.0

    total_discriminator_loss = 0.0


    # Count batches.

    total_batches = 0


    # ========================================================
    # LOOP THROUGH REAL IMAGE BATCHES
    # ========================================================

    for batch in train_loader:


        # TensorDataset returns a tuple.
        # Extract the actual image data.

        real_images = batch[0].to(

            device

        )


        # Get current batch size.
        #
        # This is useful because the last batch
        # may be smaller than BATCH_SIZE.

        current_batch_size = real_images.size(

            0

        )


        # ====================================================
        # STEP 1 : TRAIN THE DISCRIMINATOR
        # ====================================================

        # Clear old gradients.

        discriminator_optimizer.zero_grad()


        # ----------------------------------------------------
        # REAL IMAGE TRAINING
        # ----------------------------------------------------

        # Discriminator should classify
        # real images as 1.

        real_labels = torch.ones(

            current_batch_size,

            1

        ).to(

            device

        )


        # Get discriminator predictions.

        real_output = discriminator(

            real_images

        )


        # Calculate real image loss.

        real_loss = adversarial_loss(

            real_output,

            real_labels

        )


        # ----------------------------------------------------
        # FAKE IMAGE TRAINING
        # ----------------------------------------------------

        # Generate random noise.

        noise = torch.randn(

            current_batch_size,

            LATENT_DIM

        ).to(

            device

        )


        # Generate fake images.

        fake_images = generator(

            noise

        )


        # Fake images should be classified as 0.

        fake_labels = torch.zeros(

            current_batch_size,

            1

        ).to(

            device

        )


        # Detach fake images.
        #
        # This prevents Generator gradients
        # from being updated during
        # Discriminator training.

        fake_output = discriminator(

            fake_images.detach()

        )


        # Calculate fake image loss.

        fake_loss = adversarial_loss(

            fake_output,

            fake_labels

        )


        # ----------------------------------------------------
        # TOTAL DISCRIMINATOR LOSS
        # ----------------------------------------------------

        discriminator_loss = (

            real_loss

            +

            fake_loss

        ) / 2


        # Backpropagation.

        discriminator_loss.backward()


        # Update Discriminator.

        discriminator_optimizer.step()


        # ====================================================
        # STEP 2 : TRAIN THE GENERATOR
        # ====================================================

        # Clear old Generator gradients.

        generator_optimizer.zero_grad()


        # Generate new random noise.

        noise = torch.randn(

            current_batch_size,

            LATENT_DIM

        ).to(

            device

        )


        # Generate fake images.

        generated_images = generator(

            noise

        )


        # The Generator wants the Discriminator
        # to believe fake images are real.
        #
        # Therefore, target labels are 1.

        generator_labels = torch.ones(

            current_batch_size,

            1

        ).to(

            device

        )


        # Get discriminator prediction.

        generator_output = discriminator(

            generated_images

        )


        # Calculate Generator loss.

        generator_loss = adversarial_loss(

            generator_output,

            generator_labels

        )


        # Backpropagation.

        generator_loss.backward()


        # Update Generator.

        generator_optimizer.step()


        # ====================================================
        # STORE BATCH LOSSES
        # ====================================================

        total_discriminator_loss += (

            discriminator_loss.item()

        )


        total_generator_loss += (

            generator_loss.item()

        )


        total_batches += 1


    # ========================================================
    # CALCULATE AVERAGE EPOCH LOSSES
    # ========================================================

    average_discriminator_loss = (

        total_discriminator_loss

        /

        total_batches

    )


    average_generator_loss = (

        total_generator_loss

        /

        total_batches

    )


    # Save losses.

    discriminator_losses.append(

        average_discriminator_loss

    )


    generator_losses.append(

        average_generator_loss

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

            f"| Discriminator Loss: "

            f"{average_discriminator_loss:.4f} "

            f"| Generator Loss: "

            f"{average_generator_loss:.4f}"

        )


    # ========================================================
    # GENERATE IMAGES DURING TRAINING
    # ========================================================

    if (

        (epoch + 1) in DISPLAY_EPOCHS

    ):


        # Switch Generator to evaluation mode.

        generator.eval()


        # Disable gradient calculation.

        with torch.no_grad():


            # Generate images using the same
            # fixed noise every time.

            generated_samples = generator(

                fixed_noise

            )


        # Create visualization.

        fig, axes = plt.subplots(

            2,

            5,

            figsize=(10, 5)

        )


        for index in range(NUM_FIXED_SAMPLES):


            # Convert tensor to NumPy.

            image = generated_samples[

                index

            ].cpu().numpy()


            # Convert from -1,+1 back to 0,1.

            image = (

                image + 1

            ) / 2


            # Reshape into 8x8 image.

            image = image.reshape(

                IMAGE_HEIGHT,

                IMAGE_WIDTH

            )


            # Display image.

            ax = axes.flat[

                index

            ]


            ax.imshow(

                image,

                cmap="gray",

                vmin=0,

                vmax=1

            )


            ax.axis(

                "off"

            )


        plt.suptitle(

            f"Generated Digits - Epoch {epoch + 1}"

        )


        plt.tight_layout()


        plt.show()


print()

print("=" * 70)

print("GAN TRAINING COMPLETED SUCCESSFULLY")

print("=" * 70)


# ============================================================
# SECTION 6 : VISUALIZE GENERATOR AND DISCRIMINATOR LOSSES
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.plot(

    generator_losses,

    label="Generator Loss"

)


plt.plot(

    discriminator_losses,

    label="Discriminator Loss"

)


plt.title(

    "GAN Training Loss"

)


plt.xlabel(

    "Epoch"

)


plt.ylabel(

    "Loss"

)


plt.legend()


plt.grid(

    alpha=0.3

)


plt.show()


# ============================================================
# SECTION 7 : GENERATE FINAL SYNTHETIC DIGITS
# ============================================================

# Number of completely new images
# to generate.

NUM_FINAL_IMAGES = 20


# Put Generator into evaluation mode.

generator.eval()


# Create new random noise.

final_noise = torch.randn(

    NUM_FINAL_IMAGES,

    LATENT_DIM

).to(

    device

)


# Generate new synthetic images.

with torch.no_grad():


    final_generated_images = generator(

        final_noise

    )


print()

print("Final Synthetic Images Generated Successfully")

print(

    "Generated Tensor Shape:",

    final_generated_images.shape

)


# ============================================================
# SECTION 8 : VISUALIZE FINAL GENERATED IMAGES
# ============================================================

fig, axes = plt.subplots(

    4,

    5,

    figsize=(10, 8)

)


for index in range(NUM_FINAL_IMAGES):


    # Get generated image.

    image = final_generated_images[

        index

    ].cpu().numpy()


    # Convert from -1,+1 to 0,1.

    image = (

        image + 1

    ) / 2


    # Reshape into 8x8.

    image = image.reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    # Display image.

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

        f"Generated {index + 1}"

    )


    ax.axis(

        "off"

    )


plt.suptitle(

    "Final Synthetic Digits Generated by GAN"

)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 9 : CREATE REUSABLE IMAGE GENERATION FUNCTION
# ============================================================

def generate_synthetic_digits(

    number_of_images=10

):


    # Put Generator into evaluation mode.

    generator.eval()


    # Generate random noise vectors.

    noise = torch.randn(

        number_of_images,

        LATENT_DIM

    ).to(

        device

    )


    # Generate images.

    with torch.no_grad():


        synthetic_images = generator(

            noise

        )


    # Move generated images to CPU.

    synthetic_images = synthetic_images.cpu()


    return synthetic_images


print()

print("Reusable Image Generation Function Created Successfully")


# ============================================================
# SECTION 10 : TEST REUSABLE GENERATION FUNCTION
# ============================================================

test_generated_images = generate_synthetic_digits(

    number_of_images=10

)


print()

print(

    "Generated Images Shape:",

    test_generated_images.shape

)


# ============================================================
# SECTION 11 : VISUALIZE FUNCTION OUTPUT
# ============================================================

fig, axes = plt.subplots(

    2,

    5,

    figsize=(10, 5)

)


for index in range(10):


    # Convert tensor to NumPy.

    image = test_generated_images[

        index

    ].numpy()


    # Convert from -1,+1 to 0,1.

    image = (

        image + 1

    ) / 2


    # Reshape into 8x8.

    image = image.reshape(

        IMAGE_HEIGHT,

        IMAGE_WIDTH

    )


    # Display image.

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

    "Generated Images Using Reusable Function"

)


plt.tight_layout()


plt.show()


# ============================================================
# SECTION 12 : SAVE TRAINED MODELS
# ============================================================

GENERATOR_MODEL_PATH = "gan_generator_digits.pth"


DISCRIMINATOR_MODEL_PATH = "gan_discriminator_digits.pth"


# Save Generator.

torch.save(

    generator.state_dict(),

    GENERATOR_MODEL_PATH

)


# Save Discriminator.

torch.save(

    discriminator.state_dict(),

    DISCRIMINATOR_MODEL_PATH

)


print()

print("=" * 70)

print("MODELS SAVED SUCCESSFULLY")

print("=" * 70)


print()

print(

    "Generator Model:",

    GENERATOR_MODEL_PATH

)


print(

    "Discriminator Model:",

    DISCRIMINATOR_MODEL_PATH

)


# ============================================================
# SECTION 13 : FINAL GAN EVALUATION
# ============================================================

# Check final Generator and Discriminator losses.

final_generator_loss = generator_losses[-1]

final_discriminator_loss = discriminator_losses[-1]


print()

print("=" * 70)

print("FINAL GAN TRAINING RESULTS")

print("=" * 70)


print()

print(

    "Final Generator Loss:",

    round(

        final_generator_loss,

        4

    )

)


print(

    "Final Discriminator Loss:",

    round(

        final_discriminator_loss,

        4

    )

)


# ============================================================
# SECTION 14 : CREATE FINAL PROJECT SUMMARY
# ============================================================

final_project_info = {


    "Project":

        "Image Generation using GANs",


    "Dataset":

        "Scikit-learn Digits Dataset",


    "Total Images":

        len(X_tensor),


    "Image Shape":

        f"{IMAGE_HEIGHT} x {IMAGE_WIDTH}",


    "Latent Dimension":

        LATENT_DIM,


    "Batch Size":

        BATCH_SIZE,


    "Epochs":

        EPOCHS,


    "Learning Rate":

        LEARNING_RATE,


    "Final Generator Loss":

        round(

            final_generator_loss,

            4

        ),


    "Final Discriminator Loss":

        round(

            final_discriminator_loss,

            4

        ),


    "Generator Model":

        GENERATOR_MODEL_PATH,


    "Discriminator Model":

        DISCRIMINATOR_MODEL_PATH

}


print()

print("=" * 70)

print("FINAL PROJECT INFORMATION")

print("=" * 70)


print()


for key, value in final_project_info.items():


    print(

        f"{key}: {value}"

    )


# ============================================================
# SECTION 15 : FINAL PROJECT COMPLETION
# ============================================================

print()

print("=" * 70)

print("DAY 45/100 PROJECT COMPLETED SUCCESSFULLY")

print("=" * 70)


print()

print("PROJECT: IMAGE GENERATION USING GANs")


print()

print("COMPLETE PIPELINE:")


print()

print("✔ Built-in Digits Dataset")

print("✔ Exploratory Data Analysis")

print("✔ Pixel Normalization")

print("✔ GAN DataLoader")

print("✔ Generator Network")

print("✔ Discriminator Network")

print("✔ Random Latent Noise")

print("✔ Adversarial Training")

print("✔ Generator Loss")

print("✔ Discriminator Loss")

print("✔ Generated Image Visualization")

print("✔ Training Loss Visualization")

print("✔ Synthetic Digit Generation")

print("✔ Reusable Generation Function")

print("✔ Generator Model Saving")

print("✔ Discriminator Model Saving")


print()

print("END-TO-END GAN IMAGE GENERATION PROJECT COMPLETE 🚀")
