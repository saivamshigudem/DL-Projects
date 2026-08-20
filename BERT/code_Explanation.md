# ============================================================
#  DATA UNDERSTANDING & EDA
# PROJECT : SENTIMENT ANALYSIS USING BERT
# MODEL : DISTILBERT
# ============================================================


# ============================================================
# SECTION 1 : INSTALL REQUIRED LIBRARIES
# ============================================================

# Run this only once if the libraries are not installed.

!pip install datasets transformers torch pandas matplotlib scikit-learn


# ============================================================
# SECTION 2 : IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset


# ============================================================
# SECTION 3 : LOAD IMDb DATASET
# ============================================================

# Load the IMDb dataset.
# This dataset contains movie reviews with:
# 0 = Negative
# 1 = Positive



dataset = load_dataset("stanfordnlp/imdb")


# ============================================================
# SECTION 4 : VIEW DATASET STRUCTURE
# ============================================================

print(dataset)


# ============================================================
# SECTION 5 : CREATE CPU-FRIENDLY SUBSETS
# ============================================================

# Use a small subset because training BERT-family models
# on a CPU can be slow.
#
# Train      : 2000 reviews
# Validation : 500 reviews
# Test       : 500 reviews

train_data = dataset["train"].shuffle(seed=42).select(range(2000))

validation_data = dataset["test"].shuffle(seed=42).select(range(500))

test_data = dataset["test"].shuffle(seed=100).select(range(500))


# ============================================================
# SECTION 6 : CONVERT TO PANDAS DATAFRAMES
# ============================================================

# Convert Hugging Face datasets into Pandas DataFrames
# for easier EDA.

train_df = train_data.to_pandas()

validation_df = validation_data.to_pandas()

test_df = test_data.to_pandas()


# ============================================================
# SECTION 7 : DATASET SHAPE
# ============================================================

print("Training Dataset Shape :", train_df.shape)

print("Validation Dataset Shape :", validation_df.shape)

print("Testing Dataset Shape :", test_df.shape)


# ============================================================
# SECTION 8 : DISPLAY DATASET
# ============================================================

print("Training Dataset")

display(train_df.head())


# ============================================================
# SECTION 9 : UNDERSTAND LABELS
# ============================================================

# Convert numerical labels into readable sentiment labels.

label_mapping = {

    0: "Negative",

    1: "Positive"

}

print("Label Mapping")

for key, value in label_mapping.items():

    print(key, "=", value)


# ============================================================
# SECTION 10 : DISPLAY SAMPLE REVIEWS
# ============================================================

for i in range(3):

    review = train_df.iloc[i]["text"]

    label = train_df.iloc[i]["label"]

    print("=" * 70)

    print("Review Number :", i + 1)

    print("Sentiment :", label_mapping[label])

    print()

    print(review[:500])

    print()


# ============================================================
# SECTION 11 : CHECK MISSING VALUES
# ============================================================

print("Missing Values")

print(train_df.isnull().sum())


# ============================================================
# SECTION 12 : CHECK DUPLICATE REVIEWS
# ============================================================

duplicates = train_df.duplicated().sum()

print("Duplicate Reviews :", duplicates)


# ============================================================
# SECTION 13 : SENTIMENT DISTRIBUTION
# ============================================================

sentiment_counts = train_df["label"].value_counts().sort_index()

print("Sentiment Distribution")

print(sentiment_counts)


# ============================================================
# SECTION 14 : SENTIMENT DISTRIBUTION VISUALIZATION
# ============================================================

sentiment_names = [

    label_mapping[label]

    for label in sentiment_counts.index

]

plt.figure(figsize=(7, 5))

plt.bar(

    sentiment_names,

    sentiment_counts.values

)

plt.title("Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Number of Reviews")

plt.show()


# ============================================================
# SECTION 15 : REVIEW LENGTH ANALYSIS
# ============================================================

# Calculate the number of words in every review.

train_df["review_length"] = train_df["text"].apply(

    lambda review: len(review.split())

)

print(train_df["review_length"].describe())


# ============================================================
# SECTION 16 : REVIEW LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(

    train_df["review_length"],

    bins=50

)

plt.title("Review Length Distribution")

plt.xlabel("Number of Words")

plt.ylabel("Number of Reviews")

plt.show()


# ============================================================
# SECTION 17 : AVERAGE REVIEW LENGTH BY SENTIMENT
# ============================================================

average_length = train_df.groupby(

    "label"

)["review_length"].mean()

print("Average Review Length")

for label, length in average_length.items():

    print(

        label_mapping[label],

        ":",

        round(length, 2),

        "words"

    )


# ============================================================
# SECTION 18 : LONGEST AND SHORTEST REVIEW
# ============================================================

longest_review_index = train_df["review_length"].idxmax()

shortest_review_index = train_df["review_length"].idxmin()

print("Longest Review")

print(

    train_df.loc[

        longest_review_index,

        "review_length"

    ],

    "words"

)

print()

print("Shortest Review")

print(

    train_df.loc[

        shortest_review_index,

        "review_length"

    ],

    "words"

)


# ============================================================
# SECTION 19 : REMOVE EDA COLUMN
# ============================================================

# Remove review_length because it was created only
# for exploratory analysis.
#
# The original text and label columns will be used
# in Part-2.

train_df.drop(

    "review_length",

    axis=1,

    inplace=True

)


# ============================================================
# SECTION 20 : DATASET SUMMARY
# ============================================================



print("Project       : Sentiment Analysis Using DistilBERT")

print("Problem Type  : Binary Classification")

print("Training Data :", len(train_df))

print("Validation Data :", len(validation_df))

print("Testing Data  :", len(test_df))

print()

print("Classes:")

print("0 -> Negative")

print("1 -> Positive")

print()

print("CPU-Friendly Dataset Configuration")

print("Maximum Training Samples : 2000")

print("Validation Samples       : 500")

print("Testing Samples          : 500")

print()

# ============================================================
#  TEXT PREPROCESSING & BERT TOKENIZATION
# PROJECT : SENTIMENT ANALYSIS USING BERT
# MODEL : DISTILBERT
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

# Import only the additional libraries required
# for text preprocessing and tokenization.

import re
import torch

from datasets import Dataset

from transformers import AutoTokenizer

from torch.utils.data import DataLoader


# ============================================================
# SECTION 2 : TEXT CLEANING FUNCTION
# ============================================================

# IMDb reviews may contain HTML tags such as <br />.
# This function removes unnecessary HTML and
# extra spaces while keeping the actual review text.

def clean_text(text):

    text = re.sub(r"<br\s*/?>", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


# ============================================================
# SECTION 3 : APPLY TEXT CLEANING
# ============================================================

# Apply the cleaning function to all datasets.

train_df["text"] = train_df["text"].apply(clean_text)

validation_df["text"] = validation_df["text"].apply(clean_text)

test_df["text"] = test_df["text"].apply(clean_text)


print("Text Cleaning Completed Successfully")


# ============================================================
# SECTION 4 : VERIFY CLEANED TEXT
# ============================================================

print("Sample Cleaned Review:\n")

print(train_df.iloc[0]["text"][:500])


# ============================================================
# SECTION 5 : CONVERT PANDAS TO HUGGING FACE DATASET
# ============================================================

# Convert the Pandas DataFrames back into Dataset format.
# This format works efficiently with Hugging Face tokenizers.

train_dataset = Dataset.from_pandas(train_df)

validation_dataset = Dataset.from_pandas(validation_df)

test_dataset = Dataset.from_pandas(test_df)


print("Train Dataset Size :", len(train_dataset))

print("Validation Dataset Size :", len(validation_dataset))

print("Test Dataset Size :", len(test_dataset))


# ============================================================
# SECTION 6 : LOAD DISTILBERT TOKENIZER
# ============================================================

# DistilBERT tokenizer converts review text into
# numerical token IDs that the model can understand.

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME

)

print("Tokenizer Loaded Successfully")


# ============================================================
# SECTION 7 : UNDERSTAND TOKENIZATION
# ============================================================

sample_text = train_df.iloc[0]["text"]

tokens = tokenizer.tokenize(

    sample_text[:200]

)

token_ids = tokenizer.convert_tokens_to_ids(

    tokens

)

print("Original Text:\n")

print(sample_text[:200])

print("\nTokens:\n")

print(tokens[:30])

print("\nToken IDs:\n")

print(token_ids[:30])


# ============================================================
# SECTION 8 : TOKENIZATION SETTINGS
# ============================================================

# MAX_LENGTH controls the maximum number of tokens.
# Keeping it at 128 reduces CPU training time
# and memory usage.

MAX_LENGTH = 128


# ============================================================
# SECTION 9 : CREATE TOKENIZATION FUNCTION
# ============================================================

# This function:
#
# 1. Converts text into token IDs
# 2. Truncates long reviews
# 3. Pads shorter reviews
# 4. Creates attention masks

def tokenize_function(examples):

    return tokenizer(

        examples["text"],

        truncation=True,

        padding="max_length",

        max_length=MAX_LENGTH

    )


# ============================================================
# SECTION 10 : TOKENIZE TRAINING DATA
# ============================================================

train_dataset = train_dataset.map(

    tokenize_function,

    batched=True

)


# ============================================================
# SECTION 11 : TOKENIZE VALIDATION DATA
# ============================================================

validation_dataset = validation_dataset.map(

    tokenize_function,

    batched=True

)


# ============================================================
# SECTION 12 : TOKENIZE TEST DATA
# ============================================================

test_dataset = test_dataset.map(

    tokenize_function,

    batched=True

)

print("All Datasets Tokenized Successfully")


# ============================================================
# SECTION 13 : REMOVE UNNECESSARY COLUMNS
# ============================================================

# The model only needs:
#
# input_ids
# attention_mask
# label

columns_to_remove = [

    "text"

]

# Remove the Pandas index column if it exists.

for dataset_name in [

    train_dataset,

    validation_dataset,

    test_dataset

]:

    if "__index_level_0__" in dataset_name.column_names:

        columns_to_remove.append(

            "__index_level_0__"

        )

# Remove columns safely.

for column in columns_to_remove:

    if column in train_dataset.column_names:

        train_dataset = train_dataset.remove_columns(

            column

        )

    if column in validation_dataset.column_names:

        validation_dataset = validation_dataset.remove_columns(

            column

        )

    if column in test_dataset.column_names:

        test_dataset = test_dataset.remove_columns(

            column

        )


# ============================================================
# SECTION 14 : SET PYTORCH FORMAT
# ============================================================

# Convert required columns into PyTorch tensors.

columns = [

    "input_ids",

    "attention_mask",

    "label"

]

train_dataset.set_format(

    type="torch",

    columns=columns

)

validation_dataset.set_format(

    type="torch",

    columns=columns

)

test_dataset.set_format(

    type="torch",

    columns=columns

)

print("PyTorch Format Applied Successfully")


# ============================================================
# SECTION 15 : CREATE DATALOADERS
# ============================================================

# Small batch size is used because the project
# is running on a slow CPU.

BATCH_SIZE = 8


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


print("DataLoaders Created Successfully")


# ============================================================
# SECTION 16 : CHECK NUMBER OF BATCHES
# ============================================================

print()

print("Training Batches :", len(train_loader))

print("Validation Batches :", len(validation_loader))

print("Testing Batches :", len(test_loader))


# ============================================================
# SECTION 17 : INSPECT ONE BATCH
# ============================================================

batch = next(

    iter(train_loader)

)


print("Batch Keys:")

print(batch.keys())

print()

print("Input IDs Shape:")

print(batch["input_ids"].shape)

print()

print("Attention Mask Shape:")

print(batch["attention_mask"].shape)

print()

print("Labels Shape:")

print(batch["label"].shape)


# ============================================================
# SECTION 18 : DISPLAY TOKENIZED EXAMPLE
# ============================================================

example_ids = batch["input_ids"][0]

example_tokens = tokenizer.convert_ids_to_tokens(

    example_ids

)

print()

print("First 30 Tokens:")

print(example_tokens[:30])

print()

print("Label:")

print(

    batch["label"][0].item(),

    "-",

    label_mapping[batch["label"][0].item()]

)


# ============================================================
# SECTION 19 : DEVICE CHECK
# ============================================================

# Check whether GPU is available.
# If not, the model will run on CPU.

device = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)

print()

print("Running On :", device)


# ============================================================
# SECTION 20 : PART-2 SUMMARY
# ============================================================



print("Text Cleaning Completed")

print("DistilBERT Tokenizer Loaded")

print("Maximum Token Length :", MAX_LENGTH)

print("Batch Size :", BATCH_SIZE)

print("Training Samples :", len(train_dataset))

print("Validation Samples :", len(validation_dataset))

print("Testing Samples :", len(test_dataset))

print()

print("Created:")

print("✔ input_ids")

print("✔ attention_mask")

print("✔ labels")

print("✔ train_loader")

print("✔ validation_loader")

print("✔ test_loader")

print()

plt.plot(

    range(1, EPOCHS + 1),

    train_losses,

    marker="o",

    label="Training Loss"

)

plt.plot(

    range(1, EPOCHS + 1),

    validation_losses,

    marker="o",

    label="Validation Loss"

)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# SECTION 11 : PLOT VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(

    range(1, EPOCHS + 1),

    [accuracy * 100 for accuracy in validation_accuracies],

    marker="o"

)

plt.title("Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.show()


# ============================================================
# SECTION 12 : TEST MODEL
# ============================================================

# Evaluate the best saved model using unseen test data.

test_predictions = []

test_labels = []


model.eval()


with torch.no_grad():

    for batch in test_loader:


        # Move data to device.

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)


        # Forward pass.

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask

        )


        # Get predicted class.

        predictions = torch.argmax(

            outputs.logits,

            dim=1

        )


        # Store results.

        test_predictions.extend(

            predictions.cpu().numpy()

        )

        test_labels.extend(

            labels.cpu().numpy()

        )


print("Testing Completed Successfully")


# ============================================================
# SECTION 13 : CALCULATE EVALUATION METRICS
# ============================================================

test_accuracy = accuracy_score(

    test_labels,

    test_predictions

)

test_precision = precision_score(

    test_labels,

    test_predictions,

    zero_division=0

)

test_recall = recall_score(

    test_labels,

    test_predictions,

    zero_division=0

)

test_f1 = f1_score(

    test_labels,

    test_predictions,

    zero_division=0

)


print()

print("=" * 50)

print("TEST RESULTS")

print("=" * 50)

print(

    "Accuracy :",

    round(test_accuracy * 100, 2),

    "%"

)

print(

    "Precision :",

    round(test_precision * 100, 2),

    "%"

)

print(

    "Recall :",

    round(test_recall * 100, 2),

    "%"

)

print(

    "F1 Score :",

    round(test_f1 * 100, 2),

    "%"

)


# ============================================================
# SECTION 14 : CLASSIFICATION REPORT
# ============================================================

print()

print("=" * 50)

print("CLASSIFICATION REPORT")

print("=" * 50)


print(

    classification_report(

        test_labels,

        test_predictions,

        target_names=[

            "Negative",

            "Positive"

        ],

        zero_division=0

    )

)


# ============================================================
# SECTION 15 : CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    test_labels,

    test_predictions

)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.colorbar()

plt.xticks(

    [0, 1],

    ["Negative", "Positive"]

)

plt.yticks(

    [0, 1],

    ["Negative", "Positive"]

)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.title("Confusion Matrix")


# Add values inside the matrix.

for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            str(cm[i, j]),

            ha="center",

            va="center"

        )


plt.show()


# ============================================================
# SECTION 16 : CUSTOM SENTIMENT PREDICTION FUNCTION
# ============================================================

# This function accepts a custom review,
# tokenizes it,
# predicts sentiment,
# and returns confidence.

def predict_sentiment(text):


    # Tokenize input text.

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=MAX_LENGTH

    )


    # Move tensors to device.

    inputs = {

        key: value.to(device)

        for key, value in inputs.items()

    }


    # Prediction mode.

    model.eval()


    with torch.no_grad():

        outputs = model(

            **inputs

        )


        probabilities = torch.softmax(

            outputs.logits,

            dim=1

        )


        confidence, prediction = torch.max(

            probabilities,

            dim=1

        )


    # Convert numerical prediction
    # into readable sentiment.

    sentiment = label_mapping[

        prediction.item()

    ]


    return {

        "Sentiment": sentiment,

        "Confidence": round(

            confidence.item() * 100,

            2

        )

    }


# ============================================================
# SECTION 17 : TEST CUSTOM REVIEWS
# ============================================================

review_1 = """

This movie was absolutely amazing.
The acting was excellent and the story was
interesting from beginning to end.

"""


review_2 = """

This was one of the worst movies I have ever watched.
The story was boring and the acting was terrible.

"""


print()

print("Review 1 Prediction")

print(

    predict_sentiment(

        review_1

    )

)


print()

print("Review 2 Prediction")

print(

    predict_sentiment(

        review_2

    )

)


# ============================================================
# SECTION 18 : DISPLAY SAVED MODEL FILES
# ============================================================

print()

print("Saved Model Files:")

for file_name in os.listdir(MODEL_SAVE_PATH):

    print("-", file_name)


# ============================================================
# SECTION 19 : FINAL PROJECT SUMMARY
# ============================================================

print()

print("=" * 65)

print("END-TO-END SENTIMENT ANALYSIS PROJECT COMPLETED")

print("=" * 65)

print()

print("✔ IMDb Dataset Loaded")

print("✔ CPU-Friendly Dataset Sampling")

print("✔ Text Cleaning Completed")

print("✔ DistilBERT Tokenization")

print("✔ Input IDs Created")

print("✔ Attention Masks Created")

print("✔ DataLoaders Created")

print("✔ DistilBERT Fine-Tuned")

print("✔ Best Model Saved")

print("✔ Training and Validation Evaluated")

print("✔ Accuracy Calculated")

print("✔ Precision Calculated")

print("✔ Recall Calculated")

print("✔ F1 Score Calculated")

print("✔ Confusion Matrix Generated")

print("✔ Classification Report Generated")

print("✔ Custom Text Prediction Created")

print()

print("MODEL SAVE LOCATION:")

print(MODEL_SAVE_PATH)

print()

print("PROJECT COMPLETED SUCCESSFULLY!")
# ============================================================
# DISTILBERT MODEL TRAINING & EVALUATION
# PROJECT : SENTIMENT ANALYSIS USING BERT
# MODEL : DISTILBERT
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import torch
import torch.nn as nn

from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# SECTION 2 : DEFINE MODEL VARIABLES
# ============================================================

# IMPORTANT:
# MODEL_NAME = pretrained Hugging Face model
# MODEL_SAVE_PATH = local folder for your trained model

MODEL_NAME = "distilbert-base-uncased"

MODEL_SAVE_PATH = "sentiment_bert_model"

NUM_LABELS = 2


# ============================================================
# SECTION 3 : CHECK DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Running On :", device)


# ============================================================
# SECTION 4 : LOAD PRETRAINED DISTILBERT
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

model = model.to(device)

print()
print("DistilBERT Model Loaded Successfully")
print("Model Name :", MODEL_NAME)


# ============================================================
# SECTION 5 : DEFINE OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)


# ============================================================
# SECTION 6 : TRAINING SETTINGS
# ============================================================

EPOCHS = 2

total_training_steps = len(train_loader) * EPOCHS

print()
print("Epochs :", EPOCHS)

print("Training Batches :", len(train_loader))

print("Total Training Steps :", total_training_steps)


# ============================================================
# SECTION 7 : LEARNING RATE SCHEDULER
# ============================================================

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_training_steps
)


# ============================================================
# SECTION 8 : CREATE MODEL SAVE DIRECTORY
# ============================================================

os.makedirs(
    MODEL_SAVE_PATH,
    exist_ok=True
)

print()
print("Model Save Path :")
print(os.path.abspath(MODEL_SAVE_PATH))


# ============================================================
# SECTION 9 : TRAINING HISTORY
# ============================================================

train_losses = []

validation_losses = []

validation_accuracies = []


# ============================================================
# SECTION 10 : TRAINING LOOP
# ============================================================

best_validation_loss = float("inf")


for epoch in range(EPOCHS):

    print()
    print("=" * 70)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 70)


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0


    for batch_number, batch in enumerate(train_loader):

        # Move tensors to device

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)


        # Clear previous gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


        # Get loss

        loss = outputs.loss


        # Backpropagation

        loss.backward()


        # Update model parameters

        optimizer.step()


        # Update learning rate

        scheduler.step()


        # Accumulate loss

        total_train_loss += loss.item()


        # Display progress

        if (batch_number + 1) % 50 == 0:

            print(
                f"Batch {batch_number + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )


    # Average training loss

    average_train_loss = (
        total_train_loss / len(train_loader)
    )

    train_losses.append(
        average_train_loss
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    total_validation_loss = 0

    validation_predictions = []

    validation_labels = []


    with torch.no_grad():

        for batch in validation_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["label"].to(device)


            # Forward pass

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )


            # Validation loss

            loss = outputs.loss

            total_validation_loss += loss.item()


            # Predictions

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )


            validation_predictions.extend(
                predictions.cpu().numpy()
            )

            validation_labels.extend(
                labels.cpu().numpy()
            )


    # Average validation loss

    average_validation_loss = (
        total_validation_loss / len(validation_loader)
    )

    validation_losses.append(
        average_validation_loss
    )


    # Validation accuracy

    validation_accuracy = accuracy_score(
        validation_labels,
        validation_predictions
    )

    validation_accuracies.append(
        validation_accuracy
    )


    # --------------------------------------------------------
    # DISPLAY EPOCH RESULTS
    # --------------------------------------------------------

    print()

    print("Training Loss     :",
          round(average_train_loss, 4))

    print("Validation Loss   :",
          round(average_validation_loss, 4))

    print("Validation Accuracy:",
          round(validation_accuracy * 100, 2),
          "%")


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if average_validation_loss < best_validation_loss:

        best_validation_loss = average_validation_loss


        model.save_pretrained(
            MODEL_SAVE_PATH
        )


        tokenizer.save_pretrained(
            MODEL_SAVE_PATH
        )


        print()
        print("BEST MODEL SAVED")
        print(
            "Validation Loss :",
            round(best_validation_loss, 4)
        )


# ============================================================
# SECTION 11 : TRAINING COMPLETED
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print()

print("Best Validation Loss :",
      round(best_validation_loss, 4))

print()

print("Model saved at:")

print(
    os.path.abspath(MODEL_SAVE_PATH)
)


# ============================================================
# SECTION 12 : VERIFY SAVED MODEL FILES
# ============================================================

print()
print("=" * 70)
print("SAVED MODEL FILES")
print("=" * 70)

for file_name in os.listdir(MODEL_SAVE_PATH):

    print("-", file_name)


# ============================================================
# SECTION 13 : LOAD BEST SAVED MODEL
# ============================================================

# IMPORTANT:
# Load from MODEL_SAVE_PATH, NOT MODEL_NAME.

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH,
    local_files_only=True
)

model = model.to(device)

model.eval()


print()
print("Best Saved DistilBERT Model Loaded Successfully")


# ============================================================
# SECTION 14 : LOAD SAVED TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_SAVE_PATH,
    local_files_only=True
)

print("Saved Tokenizer Loaded Successfully")


# ============================================================
# SECTION 15 : PLOT TRAINING VS VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    validation_losses,
    marker="o",
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# SECTION 16 : PLOT VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    [accuracy * 100 for accuracy in validation_accuracies],
    marker="o"
)

plt.title("Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.show()


# ============================================================
# SECTION 17 : TEST MODEL
# ============================================================

test_predictions = []

test_labels = []


model.eval()


with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)


        # Forward pass

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        # Prediction

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )


        # Store predictions

        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_labels.extend(
            labels.cpu().numpy()
        )


print()
print("Testing Completed Successfully")


# ============================================================
# SECTION 18 : CALCULATE METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)


# ============================================================
# SECTION 19 : DISPLAY TEST RESULTS
# ============================================================

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    "Accuracy  :",
    round(test_accuracy * 100, 2),
    "%"
)

print(
    "Precision :",
    round(test_precision * 100, 2),
    "%"
)

print(
    "Recall    :",
    round(test_recall * 100, 2),
    "%"
)

print(
    "F1 Score  :",
    round(test_f1 * 100, 2),
    "%"
)


# ============================================================
# SECTION 20 : CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "Negative",
            "Positive"
        ],
        zero_division=0
    )
)


# ============================================================
# SECTION 21 : CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.yticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.title("Confusion Matrix")


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.show()


# ============================================================
# SECTION 22 : CUSTOM SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text):

    # Tokenize input

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )


    # Move tensors to device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # Evaluation mode

    model.eval()


    with torch.no_grad():

        outputs = model(
            **inputs
        )


        # Convert logits to probabilities

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )


        # Get highest probability

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    sentiment = label_mapping[
        prediction.item()
    ]


    return {
        "Sentiment": sentiment,
        "Confidence": round(
            confidence.item() * 100,
            2
        )
    }


# ============================================================
# SECTION 23 : TEST CUSTOM REVIEWS
# ============================================================

review_1 = """
This movie was absolutely amazing.
The acting was excellent and the story was
interesting from beginning to end.
"""


review_2 = """
This was one of the worst movies I have ever watched.
The story was boring and the acting was terrible.
"""


print()
print("=" * 60)
print("CUSTOM PREDICTIONS")
print("=" * 60)


print()

print("Review 1:")

print(
    predict_sentiment(review_1)
)


print()

print("Review 2:")

print(
    predict_sentiment(review_2)
)


# ============================================================
# SECTION 24 : FINAL PROJECT SUMMARY
# ============================================================

print()
print("=" * 70)
print("END-TO-END SENTIMENT ANALYSIS PROJECT COMPLETED")
print("=" * 70)

print()

print("✔ IMDb Dataset Loaded")

print("✔ CPU-Friendly Dataset Sampling")

print("✔ Text Cleaning Completed")

print("✔ DistilBERT Tokenization")

print("✔ Input IDs Created")

print("✔ Attention Masks Created")

print("✔ DataLoaders Created")

print("✔ DistilBERT Fine-Tuned")

print("✔ Best Model Saved")

print("✔ Validation Evaluated")

print("✔ Accuracy Calculated")

print("✔ Precision Calculated")

print("✔ Recall Calculated")

print("✔ F1 Score Calculated")

print("✔ Confusion Matrix Generated")

print("✔ Classification Report Generated")

print("✔ Custom Sentiment Prediction Created")

print()

print("MODEL SAVE LOCATION:")

print(
    os.path.abspath(MODEL_SAVE_PATH)
)

print()

print("PROJECT COMPLETED SUCCESSFULLY!")
