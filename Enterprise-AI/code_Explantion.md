# ============================================================
# ENTERPRISE AI — FINE-TUNING A SMALL LANGUAGE MODEL
# PART-1 : DATASET CREATION, EDA & PREPROCESSING
# FRAMEWORK : PYTORCH + HUGGING FACE
#
# CPU FRIENDLY
# LOW STORAGE
# NO EXTERNAL DATASET DOWNLOAD
# ============================================================


# ============================================================
# SECTION 1 : IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


print("Libraries Imported Successfully")


# ============================================================
# SECTION 2 : CREATE SMALL ENTERPRISE DATASET
# ============================================================

# We are creating a small enterprise-style dataset locally.
#
# This avoids downloading a large dataset and keeps the
# project suitable for a slow CPU and limited storage.
#
# Each example contains:
#
# category
# question
# answer


enterprise_data = [

    # --------------------------------------------------------
    # IT SUPPORT
    # --------------------------------------------------------

    {
        "category": "IT Support",
        "question": "How do I reset my corporate password?",
        "answer": "Open the corporate identity portal, select Reset Password, verify your identity using MFA, and create a new password."
    },

    {
        "category": "IT Support",
        "question": "My company laptop is not connecting to Wi-Fi. What should I do?",
        "answer": "First restart Wi-Fi on the laptop. If the issue continues, restart the laptop and reconnect to the corporate network. Contact IT support if the problem remains."
    },

    {
        "category": "IT Support",
        "question": "How can I install approved software on my company laptop?",
        "answer": "Open the company's approved software portal and request the required application. Software that requires administrative access must be approved by IT."
    },

    {
        "category": "IT Support",
        "question": "My laptop is running very slowly. What should I do?",
        "answer": "Restart the laptop, close unnecessary applications, and check available disk space. If performance remains slow, raise an IT support ticket."
    },


    # --------------------------------------------------------
    # VPN
    # --------------------------------------------------------

    {
        "category": "VPN",
        "question": "How can I connect to the corporate VPN?",
        "answer": "Open the approved corporate VPN application, enter your corporate credentials, complete MFA verification, and connect to the corporate VPN."
    },

    {
        "category": "VPN",
        "question": "My VPN connection keeps disconnecting. What should I do?",
        "answer": "Check your internet connection and reconnect to the VPN. If the issue continues, restart the VPN client and contact IT support."
    },

    {
        "category": "VPN",
        "question": "Can I access internal applications without the VPN?",
        "answer": "Internal applications normally require an approved corporate network or VPN connection. Connect to the corporate VPN before accessing restricted applications."
    },


    # --------------------------------------------------------
    # HR
    # --------------------------------------------------------

    {
        "category": "HR",
        "question": "How do I apply for leave?",
        "answer": "Open the employee HR portal, navigate to the Leave section, select the required leave type and dates, and submit the request for approval."
    },

    {
        "category": "HR",
        "question": "Where can I view my payslips?",
        "answer": "Log in to the employee HR portal and open the Payroll or Payslips section to view and download available payslips."
    },

    {
        "category": "HR",
        "question": "How can I update my personal information?",
        "answer": "Open the employee HR portal and navigate to your profile section. Update the permitted information and submit the changes."
    },

    {
        "category": "HR",
        "question": "How can I check my leave balance?",
        "answer": "Log in to the employee HR portal and open the Leave section to view your available and used leave balance."
    },


    # --------------------------------------------------------
    # SECURITY
    # --------------------------------------------------------

    {
        "category": "Security",
        "question": "What should I do if I receive a suspicious email?",
        "answer": "Do not click links or open attachments in the suspicious email. Report the message using the company's approved security reporting process."
    },

    {
        "category": "Security",
        "question": "Why should I use multi-factor authentication?",
        "answer": "Multi-factor authentication adds an additional verification step and helps protect corporate accounts even if a password is compromised."
    },

    {
        "category": "Security",
        "question": "Can I share my corporate password with a colleague?",
        "answer": "No. Corporate passwords must remain confidential and should never be shared with colleagues or other individuals."
    },


    # --------------------------------------------------------
    # ACCESS MANAGEMENT
    # --------------------------------------------------------

    {
        "category": "Access Management",
        "question": "How do I request access to an internal application?",
        "answer": "Submit an access request through the approved corporate access management portal. The request may require manager or application-owner approval."
    },

    {
        "category": "Access Management",
        "question": "My application access was removed. What should I do?",
        "answer": "Check whether your access has expired. If you still require access, submit a new request through the corporate access management process."
    },

    {
        "category": "Access Management",
        "question": "How long does an application access request take?",
        "answer": "Processing time depends on the application and required approvals. You can check the request status through the corporate access management portal."
    },


    # --------------------------------------------------------
    # GENERAL ENTERPRISE
    # --------------------------------------------------------

    {
        "category": "General",
        "question": "How do I create an IT support ticket?",
        "answer": "Open the corporate IT service portal, select Create Ticket, provide the issue details, and submit the ticket."
    },

    {
        "category": "General",
        "question": "How can I check the status of my support ticket?",
        "answer": "Log in to the corporate IT service portal and open the My Tickets section to view the current status of your requests."
    },

    {
        "category": "General",
        "question": "Who should I contact for an urgent IT issue?",
        "answer": "For urgent IT issues, follow the company's approved incident escalation process or contact the designated IT support team."
    }

]


# ============================================================
# SECTION 3 : CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(

    enterprise_data

)


print()

print("Enterprise Dataset Created Successfully")


# ============================================================
# SECTION 4 : DISPLAY DATASET
# ============================================================

print()

print("=" * 70)

print("ENTERPRISE DATASET")

print("=" * 70)

print()

display(df)


# ============================================================
# SECTION 5 : DATASET SHAPE
# ============================================================

print()

print("=" * 70)

print("DATASET SHAPE")

print("=" * 70)

print()

print("Number of Records:", len(df))

print("Number of Columns:", len(df.columns))


# ============================================================
# SECTION 6 : COLUMN INFORMATION
# ============================================================

print()

print("=" * 70)

print("COLUMN INFORMATION")

print("=" * 70)

print()

print(df.info())


# ============================================================
# SECTION 7 : CHECK MISSING VALUES
# ============================================================

print()

print("=" * 70)

print("MISSING VALUE ANALYSIS")

print("=" * 70)

print()

missing_values = df.isnull().sum()

print(missing_values)

print()

print(

    "Total Missing Values:",

    missing_values.sum()

)


# ============================================================
# SECTION 8 : CHECK DUPLICATE RECORDS
# ============================================================

print()

print("=" * 70)

print("DUPLICATE RECORD ANALYSIS")

print("=" * 70)

print()

duplicate_count = df.duplicated().sum()

print(

    "Duplicate Records:",

    duplicate_count

)


# ============================================================
# SECTION 9 : CHECK DATASET CATEGORIES
# ============================================================

print()

print("=" * 70)

print("ENTERPRISE CATEGORIES")

print("=" * 70)

print()

category_counts = df["category"].value_counts()

print(category_counts)


# ============================================================
# SECTION 10 : VISUALIZE CATEGORY DISTRIBUTION
# ============================================================

import matplotlib.pyplot as plt


plt.figure(

    figsize=(10, 5)

)


category_counts.plot(

    kind="bar"

)


plt.title(

    "Enterprise Dataset Category Distribution"

)


plt.xlabel(

    "Category"

)


plt.ylabel(

    "Number of Examples"

)


plt.xticks(

    rotation=30,

    ha="right"

)


plt.grid(

    axis="y",

    alpha=0.3

)


plt.tight_layout()

plt.show()


# ============================================================
# SECTION 11 : CHECK QUESTION LENGTH
# ============================================================

df["question_length"] = df["question"].apply(

    lambda x: len(x.split())

)


df["answer_length"] = df["answer"].apply(

    lambda x: len(x.split())

)


print()

print("=" * 70)

print("TEXT LENGTH ANALYSIS")

print("=" * 70)

print()

print(

    "Average Question Length:",

    round(

        df["question_length"].mean(),

        2

    ),

    "words"

)


print(

    "Average Answer Length:",

    round(

        df["answer_length"].mean(),

        2

    ),

    "words"

)


print(

    "Maximum Question Length:",

    df["question_length"].max(),

    "words"

)


print(

    "Maximum Answer Length:",

    df["answer_length"].max(),

    "words"

)


# ============================================================
# SECTION 12 : DISPLAY TEXT LENGTH INFORMATION
# ============================================================

display(

    df[

        [

            "category",

            "question_length",

            "answer_length"

        ]

    ]

)


# ============================================================
# SECTION 13 : REMOVE DUPLICATES IF ANY
# ============================================================

df = df.drop_duplicates().reset_index(

    drop=True

)


print()

print(

    "Dataset after duplicate removal:",

    len(df),

    "records"

)


# ============================================================
# SECTION 14 : TRAIN / VALIDATION SPLIT
# ============================================================

# We keep the validation set small because this is a
# CPU/storage constrained demonstration project.

train_df, validation_df = train_test_split(

    df,

    test_size=0.20,

    random_state=42,

    shuffle=True

)


# Reset indexes.

train_df = train_df.reset_index(

    drop=True

)


validation_df = validation_df.reset_index(

    drop=True

)


print()

print("=" * 70)

print("TRAIN / VALIDATION SPLIT")

print("=" * 70)

print()

print(

    "Training Examples:",

    len(train_df)

)

print(

    "Validation Examples:",

    len(validation_df)

)


# ============================================================
# SECTION 15 : CREATE INSTRUCTION FORMAT
# ============================================================

# Fine-tuning a language model requires text examples.
#
# We convert:
#
# Category
# Question
# Answer
#
# into:
#
# Instruction
# Input
# Response
#
# This format will be consumed by the tokenizer
# in Part-2.


def create_instruction_format(row):


    formatted_text = (

        "### Enterprise Support Task\n\n"

        "Category: "

        + str(row["category"])

        + "\n\n"

        "Question: "

        + str(row["question"])

        + "\n\n"

        "Answer: "

        + str(row["answer"])

    )


    return formatted_text


# Apply formatting.

train_df["text"] = train_df.apply(

    create_instruction_format,

    axis=1

)


validation_df["text"] = validation_df.apply(

    create_instruction_format,

    axis=1

)


# ============================================================
# SECTION 16 : DISPLAY SAMPLE TRAINING RECORD
# ============================================================

print()

print("=" * 70)

print("SAMPLE FINE-TUNING RECORD")

print("=" * 70)

print()

print(

    train_df["text"].iloc[0]

)


# ============================================================
# SECTION 17 : DISPLAY SAMPLE VALIDATION RECORD
# ============================================================

print()

print("=" * 70)

print("SAMPLE VALIDATION RECORD")

print("=" * 70)

print()

print(

    validation_df["text"].iloc[0]

)


# ============================================================
# SECTION 18 : CREATE FINAL TRAINING DATA
# ============================================================

# Only keep the text column for model training.

train_texts = train_df[

    ["text"]

].copy()


validation_texts = validation_df[

    ["text"]

].copy()


print()

print("=" * 70)

print("FINAL TRAINING DATA")

print("=" * 70)

print()

print(

    "Training Records:",

    len(train_texts)

)

print(

    "Validation Records:",

    len(validation_texts)

)


# ============================================================
# SECTION 19 : CHECK FINAL DATA
# ============================================================

print()

print("=" * 70)

print("FINAL DATA QUALITY CHECK")

print("=" * 70)

print()

print(

    "Training Missing Values:",

    train_texts.isnull().sum().sum()

)

print(

    "Validation Missing Values:",

    validation_texts.isnull().sum().sum()

)

print(

    "Training Duplicate Texts:",

    train_texts["text"].duplicated().sum()

)

print(

    "Validation Duplicate Texts:",

    validation_texts["text"].duplicated().sum()

)


# ============================================================
# SECTION 20 : STORE IMPORTANT CONFIGURATION
# ============================================================

# These variables will be reused in Parts 2, 3 and 4.

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_LENGTH = 128


print()

print("=" * 70)

print("PROJECT CONFIGURATION")

print("=" * 70)

print()

print("Random State:", RANDOM_STATE)

print("Validation Size:", TEST_SIZE)

print("Maximum Token Length:", MAX_LENGTH)


# ============================================================
# SECTION 21 : FINAL PART-1 SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-1 SUMMARY")

print("=" * 70)

print()

print("Project              : Enterprise AI Fine-Tuning")

print("Dataset Type         : Custom Enterprise Support Dataset")

print("Total Examples       :", len(df))

print("Training Examples    :", len(train_df))

print("Validation Examples  :", len(validation_df))

print("Categories           :", df["category"].nunique())

print("Missing Values       :", df.isnull().sum().sum())

print("Duplicate Records    :", df.duplicated().sum())

print("Max Token Length     :", MAX_LENGTH)


# ============================================================
# SECTION 22 : IMPORTANT VARIABLES
# ============================================================

print()

print("=" * 70)

print("IMPORTANT VARIABLES CREATED")

print("=" * 70)

print()

print("✔ df")

print("✔ train_df")

print("✔ validation_df")

print("✔ train_texts")

print("✔ validation_texts")

print("✔ RANDOM_STATE")

print("✔ TEST_SIZE")

print("✔ MAX_LENGTH")


# ============================================================
# SECTION 23 : PART-1 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-1 COMPLETED SUCCESSFULLY 🚀")

print("=" * 70)

print()

print(

    "READY FOR PART-2:"

)

print(

    "SMALL PRETRAINED MODEL + TOKENIZER + LoRA/PEFT SETUP"

)
# ============================================================
# ENTERPRISE AI — FINE-TUNING A SMALL LANGUAGE MODEL
# PART-2 : PRETRAINED MODEL + TOKENIZER + LoRA SETUP
#
# CPU FRIENDLY
# LOW STORAGE
# PARAMETER-EFFICIENT FINE-TUNING
# ============================================================


# ============================================================
# SECTION 1 : INSTALL REQUIRED LIBRARIES
# ============================================================

# Run this cell only once if these packages are not installed.
#
# IMPORTANT:
# After installation, restart the Jupyter kernel if required.


!pip install -q transformers datasets peft accelerate


print("Required Libraries Installed / Available")


# ============================================================
# SECTION 2 : IMPORT LIBRARIES
# ============================================================

import os

import torch

from datasets import Dataset

from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM

)

from peft import (

    LoraConfig,

    get_peft_model,

    TaskType

)


print()

print("Libraries Imported Successfully")


# ============================================================
# SECTION 3 : CHECK DEVICE
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
# SECTION 4 : DEFINE SMALL MODEL
# ============================================================

# We deliberately use a small GPT-2 model.
#
# This is much more suitable for your CPU and storage
# limitations than a multi-billion-parameter LLM.
#
# Model:
#
# distilgpt2
#
# It is a smaller GPT-2 architecture and is suitable
# for demonstrating the enterprise fine-tuning workflow.


MODEL_NAME = "distilgpt2"


print()

print("=" * 70)

print("MODEL CONFIGURATION")

print("=" * 70)

print()

print("Base Model:", MODEL_NAME)

print("Maximum Sequence Length:", MAX_LENGTH)


# ============================================================
# SECTION 5 : LOAD TOKENIZER
# ============================================================

print()

print("Loading Tokenizer...")


tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME

)


print()

print("Tokenizer Loaded Successfully")


# ============================================================
# SECTION 6 : SET PADDING TOKEN
# ============================================================

# GPT-style tokenizers may not have a dedicated
# padding token.
#
# We use the EOS token as the padding token.


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


print()

print("Padding Token Configured")

print(

    "Pad Token ID:",

    tokenizer.pad_token_id

)


# ============================================================
# SECTION 7 : CREATE HUGGING FACE TRAINING DATASET
# ============================================================

# Part-1 already created:
#
# train_texts
# validation_texts
#
# Now we convert them into Hugging Face Dataset objects.


train_dataset = Dataset.from_pandas(

    train_texts,

    preserve_index=False

)


validation_dataset = Dataset.from_pandas(

    validation_texts,

    preserve_index=False

)


print()

print("=" * 70)

print("HUGGING FACE DATASETS CREATED")

print("=" * 70)

print()

print(

    "Training Examples:",

    len(train_dataset)

)

print(

    "Validation Examples:",

    len(validation_dataset)

)


# ============================================================
# SECTION 8 : TOKENIZATION FUNCTION
# ============================================================

def tokenize_function(

    examples

):


    # Convert enterprise text into token IDs.

    tokenized = tokenizer(

        examples["text"],

        truncation=True,

        padding="max_length",

        max_length=MAX_LENGTH

    )


    # For causal language modeling:
    #
    # labels are the same token sequence
    # as input_ids.
    #
    # The model learns to predict the next token.


    tokenized["labels"] = [

        ids.copy()

        for ids in tokenized["input_ids"]

    ]


    return tokenized


print()

print("Tokenization Function Created Successfully")


# ============================================================
# SECTION 9 : TOKENIZE TRAINING DATA
# ============================================================

print()

print("Tokenizing Training Dataset...")


tokenized_train_dataset = train_dataset.map(

    tokenize_function,

    batched=True,

    remove_columns=train_dataset.column_names

)


print()

print("Training Dataset Tokenized Successfully")


# ============================================================
# SECTION 10 : TOKENIZE VALIDATION DATA
# ============================================================

print()

print("Tokenizing Validation Dataset...")


tokenized_validation_dataset = validation_dataset.map(

    tokenize_function,

    batched=True,

    remove_columns=validation_dataset.column_names

)


print()

print("Validation Dataset Tokenized Successfully")


# ============================================================
# SECTION 11 : CHECK TOKENIZED DATA
# ============================================================

print()

print("=" * 70)

print("TOKENIZATION CHECK")

print("=" * 70)

print()

print(

    "Input IDs Shape:",

    len(

        tokenized_train_dataset[0]["input_ids"]

    )

)

print(

    "Attention Mask Shape:",

    len(

        tokenized_train_dataset[0]["attention_mask"]

    )

)

print(

    "Labels Shape:",

    len(

        tokenized_train_dataset[0]["labels"]

    )

)


# ============================================================
# SECTION 12 : DECODE A TOKENIZED EXAMPLE
# ============================================================

sample_input_ids = tokenized_train_dataset[0][

    "input_ids"

]


decoded_text = tokenizer.decode(

    sample_input_ids,

    skip_special_tokens=True

)


print()

print("=" * 70)

print("DECODED TOKENIZED EXAMPLE")

print("=" * 70)

print()

print(decoded_text)


# ============================================================
# SECTION 13 : LOAD PRETRAINED GPT MODEL
# ============================================================

print()

print("=" * 70)

print("LOADING PRETRAINED MODEL")

print("=" * 70)

print()

print("Model:", MODEL_NAME)

print("Please wait...")


model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME

)


print()

print("Pretrained Model Loaded Successfully")


# ============================================================
# SECTION 14 : CONFIGURE MODEL PADDING
# ============================================================

model.config.pad_token_id = tokenizer.pad_token_id


print()

print("Model Padding Configuration Updated")


# ============================================================
# SECTION 15 : CHECK BASE MODEL PARAMETERS
# ============================================================

total_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

)


trainable_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad

)


print()

print("=" * 70)

print("BASE MODEL PARAMETERS")

print("=" * 70)

print()

print(

    "Total Parameters:",

    f"{total_parameters:,}"

)

print(

    "Trainable Parameters:",

    f"{trainable_parameters:,}"

)


# ============================================================
# SECTION 16 : CREATE LoRA CONFIGURATION
# ============================================================

# LoRA does NOT update all model parameters.
#
# Instead, it adds small trainable matrices
# to selected model layers.
#
# This significantly reduces the number of
# trainable parameters.


lora_config = LoraConfig(

    task_type=TaskType.CAUSAL_LM,

    r=4,

    lora_alpha=8,

    lora_dropout=0.05,

    target_modules=[

        "c_attn"

    ],

    bias="none"

)


print()

print("=" * 70)

print("LoRA CONFIGURATION")

print("=" * 70)

print()

print("Rank (r):", 4)

print("LoRA Alpha:", 8)

print("LoRA Dropout:", 0.05)

print("Target Module:", "c_attn")


# ============================================================
# SECTION 17 : APPLY LoRA TO MODEL
# ============================================================

print()

print("Applying LoRA...")


model = get_peft_model(

    model,

    lora_config

)


print()

print("LoRA Applied Successfully")


# ============================================================
# SECTION 18 : DISPLAY LoRA TRAINABLE PARAMETERS
# ============================================================

print()

print("=" * 70)

print("LoRA PARAMETER INFORMATION")

print("=" * 70)

print()


model.print_trainable_parameters()


# ============================================================
# SECTION 19 : CALCULATE TRAINABLE PARAMETER RATIO
# ============================================================

total_parameters_after_lora = sum(

    parameter.numel()

    for parameter in model.parameters()

)


trainable_parameters_after_lora = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad

)


trainable_percentage = (

    trainable_parameters_after_lora

    /

    total_parameters_after_lora

) * 100


print()

print(

    "Total Parameters:",

    f"{total_parameters_after_lora:,}"

)


print(

    "Trainable Parameters:",

    f"{trainable_parameters_after_lora:,}"

)


print(

    "Trainable Percentage:",

    f"{trainable_percentage:.4f}%"

)


# ============================================================
# SECTION 20 : MOVE MODEL TO DEVICE
# ============================================================

model = model.to(

    device

)


print()

print("LoRA Model Moved to:", device)


# ============================================================
# SECTION 21 : CHECK MODEL TRAINING MODE
# ============================================================

model.train()


print()

print("Model Training Mode Enabled")


# ============================================================
# SECTION 22 : TEST TOKENIZATION + MODEL FORWARD PASS
# ============================================================

# Before Part-3 training, we perform a small
# forward-pass test.
#
# This helps identify configuration issues
# before starting CPU-intensive training.


sample = tokenized_train_dataset[0]


sample_input_ids = torch.tensor(

    [

        sample["input_ids"]

    ],

    dtype=torch.long

).to(

    device

)


sample_attention_mask = torch.tensor(

    [

        sample["attention_mask"]

    ],

    dtype=torch.long

).to(

    device

)


print()

print("=" * 70)

print("FORWARD PASS TEST")

print("=" * 70)

print()


with torch.no_grad():

    output = model(

        input_ids=sample_input_ids,

        attention_mask=sample_attention_mask

    )


print(

    "Input Shape:",

    sample_input_ids.shape

)


print(

    "Logits Shape:",

    output.logits.shape

)


print()

print("Forward Pass Successful")


# ============================================================
# SECTION 23 : SAVE LoRA CONFIGURATION INFORMATION
# ============================================================

# We do NOT save the full base model here.
#
# This keeps storage usage low.
#
# Part-3 will perform training.
#
# Part-4 will save the final LoRA adapter.


MODEL_OUTPUT_DIR = "enterprise_lora_model"


print()

print("=" * 70)

print("OUTPUT CONFIGURATION")

print("=" * 70)

print()

print(

    "Output Directory:",

    MODEL_OUTPUT_DIR

)


# ============================================================
# SECTION 24 : FINAL PART-2 SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-2 SUMMARY")

print("=" * 70)

print()

print("Base Model             :", MODEL_NAME)

print("Tokenizer              :", MODEL_NAME)

print("Maximum Length         :", MAX_LENGTH)

print("Training Examples      :", len(tokenized_train_dataset))

print("Validation Examples    :", len(tokenized_validation_dataset))

print("LoRA Rank              :", 4)

print("LoRA Alpha             :", 8)

print("LoRA Dropout           :", 0.05)

print("Trainable Parameters   :", f"{trainable_parameters_after_lora:,}")

print("Trainable Percentage   :", f"{trainable_percentage:.4f}%")

print("Device                 :", device)


# ============================================================
# SECTION 25 : IMPORTANT VARIABLES CREATED
# ============================================================

print()

print("=" * 70)

print("IMPORTANT VARIABLES CREATED")

print("=" * 70)

print()

print("✔ tokenizer")

print("✔ train_dataset")

print("✔ validation_dataset")

print("✔ tokenized_train_dataset")

print("✔ tokenized_validation_dataset")

print("✔ model")

print("✔ lora_config")

print("✔ MODEL_NAME")

print("✔ MODEL_OUTPUT_DIR")


# ============================================================
# SECTION 26 : PART-2 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-2 COMPLETED SUCCESSFULLY 🚀")

print("=" * 70)

print()

print("READY FOR PART-3")

print()

print("NEXT:")

print("LoRA FINE-TUNING + TRAINING + VALIDATION")
# ============================================================
# ENTERPRISE AI — FINE-TUNING A SMALL LANGUAGE MODEL
# PART-3 : LoRA FINE-TUNING + TRAINING + VALIDATION
#
# CONTINUES FROM PART-2
# CPU FRIENDLY
# ============================================================


# ============================================================
# SECTION 1 : IMPORT TRAINING LIBRARIES
# ============================================================

from transformers import (

    Trainer,

    TrainingArguments,

    DataCollatorForLanguageModeling

)

import torch


print("Part-3 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : VERIFY IMPORTANT VARIABLES FROM PART-2
# ============================================================

print()

print("=" * 70)

print("VERIFYING PART-2 COMPONENTS")

print("=" * 70)

print()


required_variables = [

    "model",

    "tokenizer",

    "tokenized_train_dataset",

    "tokenized_validation_dataset",

    "MODEL_OUTPUT_DIR",

    "MAX_LENGTH"

]


for variable_name in required_variables:

    if variable_name in globals():

        print(

            "✔",

            variable_name,

            "available"

        )

    else:

        print(

            "✘",

            variable_name,

            "NOT FOUND"

        )


# ============================================================
# SECTION 3 : CHECK DEVICE
# ============================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print()

print("Training Device:", device)


# ============================================================
# SECTION 4 : CONFIGURE DATA COLLATOR
# ============================================================

# Data collator prepares batches for the language model.
#
# mlm=False means:
#
# We are training a Causal Language Model.
#
# GPT-style models predict the next token.


data_collator = DataCollatorForLanguageModeling(

    tokenizer=tokenizer,

    mlm=False

)


print()

print("Data Collator Created Successfully")


# ============================================================
# SECTION 5 : DEFINE CPU-FRIENDLY TRAINING CONFIGURATION
# ============================================================

# Keep these values small because your CPU is slow.
#
# One training epoch is enough to demonstrate the complete
# fine-tuning pipeline.
#
# You can increase EPOCHS later if your system handles it.


EPOCHS = 1


TRAIN_BATCH_SIZE = 1


EVAL_BATCH_SIZE = 1


LEARNING_RATE = 2e-4


GRADIENT_ACCUMULATION_STEPS = 4


# ============================================================
# SECTION 6 : DEFINE TRAINING OUTPUT DIRECTORY
# ============================================================

TRAINING_OUTPUT_DIR = (

    "enterprise_finetuning_output"

)


print()

print("=" * 70)

print("TRAINING CONFIGURATION")

print("=" * 70)

print()

print("Epochs:", EPOCHS)

print("Training Batch Size:", TRAIN_BATCH_SIZE)

print("Evaluation Batch Size:", EVAL_BATCH_SIZE)

print("Learning Rate:", LEARNING_RATE)

print(

    "Gradient Accumulation:",

    GRADIENT_ACCUMULATION_STEPS

)

print("Output Directory:", TRAINING_OUTPUT_DIR)


# ============================================================
# SECTION 7 : CREATE TRAINING ARGUMENTS
# ============================================================

training_arguments = TrainingArguments(

    output_dir=TRAINING_OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    logging_steps=5,

    save_strategy="epoch",

    eval_strategy="epoch",

    report_to="none",

    fp16=False,

    dataloader_num_workers=0,

    remove_unused_columns=False,

    save_total_limit=1,

    logging_first_step=True

)


print()

print("TrainingArguments Created Successfully")


# ============================================================
# SECTION 8 : CREATE TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    processing_class=tokenizer,  # Replaced 'tokenizer' with 'processing_class'
    data_collator=data_collator
)


print()

print("=" * 70)

print("HUGGING FACE TRAINER CREATED")

print("=" * 70)

print()

print("Training Dataset:", len(tokenized_train_dataset))

print("Validation Dataset:", len(tokenized_validation_dataset))


# ============================================================
# SECTION 9 : DISPLAY TRAINABLE PARAMETERS
# ============================================================

print()

print("=" * 70)

print("TRAINABLE PARAMETERS BEFORE TRAINING")

print("=" * 70)

print()


try:

    model.print_trainable_parameters()

except Exception:

    trainable_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )

    total_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()

    )

    print(

        "Trainable Parameters:",

        f"{trainable_parameters:,}"

    )

    print(

        "Total Parameters:",

        f"{total_parameters:,}"

    )


# ============================================================
# SECTION 10 : START LoRA FINE-TUNING
# ============================================================

print()

print("=" * 70)

print("STARTING ENTERPRISE LoRA FINE-TUNING")

print("=" * 70)

print()

print("Please wait...")

print("CPU training may take some time.")


training_result = trainer.train()


print()

print("=" * 70)

print("FINE-TUNING COMPLETED")

print("=" * 70)


# ============================================================
# SECTION 11 : DISPLAY TRAINING METRICS
# ============================================================

print()

print("=" * 70)

print("TRAINING METRICS")

print("=" * 70)

print()


if hasattr(

    training_result,

    "metrics"

):

    for key, value in training_result.metrics.items():

        print(

            f"{key}: {value}"

        )


# ============================================================
# SECTION 12 : EVALUATE FINE-TUNED MODEL
# ============================================================

print()

print("=" * 70)

print("VALIDATING FINE-TUNED MODEL")

print("=" * 70)

print()


evaluation_results = trainer.evaluate()


print()

print("Validation Results:")

print()


for key, value in evaluation_results.items():

    print(

        f"{key}: {value}"

    )


# ============================================================
# SECTION 13 : EXTRACT VALIDATION LOSS
# ============================================================

validation_loss = evaluation_results.get(

    "eval_loss",

    None

)


print()

if validation_loss is not None:

    print(

        "Final Validation Loss:",

        round(

            validation_loss,

            6

        )

    )


# ============================================================
# SECTION 14 : EXTRACT TRAINING LOG HISTORY
# ============================================================

log_history = trainer.state.log_history


print()

print("=" * 70)

print("TRAINING LOG HISTORY")

print("=" * 70)

print()


for log in log_history:

    print(log)


# ============================================================
# SECTION 15 : CREATE LOSS HISTORY
# ============================================================

training_loss_values = []

validation_loss_values = []

training_steps = []

validation_steps = []


for log in log_history:


    if "loss" in log and "eval_loss" not in log:

        training_loss_values.append(

            log["loss"]

        )

        training_steps.append(

            log.get(

                "step",

                len(training_steps) + 1

            )

        )


    if "eval_loss" in log:

        validation_loss_values.append(

            log["eval_loss"]

        )

        validation_steps.append(

            log.get(

                "step",

                len(validation_steps) + 1

            )

        )


# ============================================================
# SECTION 16 : VISUALIZE TRAINING LOSS
# ============================================================

import matplotlib.pyplot as plt


if len(training_loss_values) > 0:


    plt.figure(

        figsize=(10, 5)

    )


    plt.plot(

        training_steps,

        training_loss_values,

        marker="o"

    )


    plt.title(

        "LoRA Fine-Tuning Training Loss"

    )


    plt.xlabel(

        "Training Step"

    )


    plt.ylabel(

        "Loss"

    )


    plt.grid(

        alpha=0.3

    )


    plt.tight_layout()


    plt.show()


else:

    print(

        "Training loss history is not available."

    )


# ============================================================
# SECTION 17 : VISUALIZE VALIDATION LOSS
# ============================================================

if len(validation_loss_values) > 0:


    plt.figure(

        figsize=(10, 5)

    )


    plt.plot(

        validation_steps,

        validation_loss_values,

        marker="o"

    )


    plt.title(

        "Validation Loss"

    )


    plt.xlabel(

        "Training Step"

    )


    plt.ylabel(

        "Validation Loss"

    )


    plt.grid(

        alpha=0.3

    )


    plt.tight_layout()


    plt.show()


else:

    print(

        "Validation loss history is not available."

    )


# ============================================================
# SECTION 18 : CALCULATE PERPLEXITY
# ============================================================

# Perplexity is commonly used for language models.
#
# Formula:
#
# perplexity = exp(loss)
#
# Lower perplexity generally indicates that
# the model predicts the evaluation text better.


if validation_loss is not None:


    try:

        perplexity = np.exp(

            validation_loss

        )


        print()

        print("=" * 70)

        print("LANGUAGE MODEL METRICS")

        print("=" * 70)

        print()

        print(

            "Validation Loss:",

            round(

                validation_loss,

                6

            )

        )


        print(

            "Perplexity:",

            round(

                perplexity,

                4

            )

        )


    except:

        print(

            "Perplexity could not be calculated."

        )


# ============================================================
# SECTION 19 : SAVE LoRA ADAPTER
# ============================================================

# IMPORTANT:
#
# We save only the LoRA adapter.
#
# We do NOT create another complete copy of the
# pretrained model.
#
# This keeps storage usage much lower.


FINAL_MODEL_PATH = (

    "enterprise_lora_adapter"

)


print()

print("=" * 70)

print("SAVING LoRA ADAPTER")

print("=" * 70)

print()


model.save_pretrained(

    FINAL_MODEL_PATH

)


tokenizer.save_pretrained(

    FINAL_MODEL_PATH

)


print(

    "LoRA Adapter Saved Successfully"

)


print(

    "Saved Location:",

    FINAL_MODEL_PATH

)


# ============================================================
# SECTION 20 : TEST MODEL GENERATION
# ============================================================

print()

print("=" * 70)

print("TESTING FINE-TUNED MODEL")

print("=" * 70)


# Put model into evaluation mode.

model.eval()


# Example enterprise question.

test_question = (

    "### Enterprise Support Task\n\n"

    "Category: IT Support\n\n"

    "Question: How do I reset my corporate password?\n\n"

    "Answer:"

)


# Tokenize the question.

inputs = tokenizer(

    test_question,

    return_tensors="pt",

    truncation=True,

    max_length=MAX_LENGTH

)


# Move tensors to model device.

inputs = {

    key: value.to(device)

    for key, value in inputs.items()

}


print()

print("Enterprise Question:")

print()

print(

    "How do I reset my corporate password?"

)


print()

print("Generating Response...")


# Generate model response.

with torch.no_grad():

    generated_output = model.generate(

        **inputs,

        max_new_tokens=40,

        do_sample=False,

        pad_token_id=tokenizer.pad_token_id

    )


# Decode output.

generated_text = tokenizer.decode(

    generated_output[0],

    skip_special_tokens=True

)


print()

print("=" * 70)

print("MODEL RESPONSE")

print("=" * 70)

print()

print(generated_text)


# ============================================================
# SECTION 21 : TEST SECOND ENTERPRISE QUESTION
# ============================================================

second_question = (

    "### Enterprise Support Task\n\n"

    "Category: VPN\n\n"

    "Question: How can I connect to the corporate VPN?\n\n"

    "Answer:"

)


second_inputs = tokenizer(

    second_question,

    return_tensors="pt",

    truncation=True,

    max_length=MAX_LENGTH

)


second_inputs = {

    key: value.to(device)

    for key, value in second_inputs.items()

}


with torch.no_grad():

    second_output = model.generate(

        **second_inputs,

        max_new_tokens=40,

        do_sample=False,

        pad_token_id=tokenizer.pad_token_id

    )


second_generated_text = tokenizer.decode(

    second_output[0],

    skip_special_tokens=True

)


print()

print("=" * 70)

print("SECOND ENTERPRISE TEST")

print("=" * 70)

print()

print(

    "Question: How can I connect to the corporate VPN?"

)

print()

print("Response:")

print()

print(second_generated_text)


# ============================================================
# SECTION 22 : FINAL PART-3 SUMMARY
# ============================================================

print()

print("=" * 70)

print("PART-3 SUMMARY")

print("=" * 70)

print()

print("Base Model             :", MODEL_NAME)

print("Fine-Tuning Method     :", "LoRA / PEFT")

print("Epochs                 :", EPOCHS)

print("Training Batch Size    :", TRAIN_BATCH_SIZE)

print("Learning Rate          :", LEARNING_RATE)

print("Training Examples      :", len(tokenized_train_dataset))

print("Validation Examples    :", len(tokenized_validation_dataset))


if validation_loss is not None:

    print(

        "Validation Loss       :",

        round(

            validation_loss,

            6

        )

    )


print(

    "Saved Adapter          :",

    FINAL_MODEL_PATH

)

print(

    "Device                 :",

    device

)


# ============================================================
# SECTION 23 : IMPORTANT VARIABLES CREATED
# ============================================================

print()

print("=" * 70)

print("IMPORTANT VARIABLES CREATED")

print("=" * 70)

print()

print("✔ trainer")

print("✔ training_arguments")

print("✔ data_collator")

print("✔ training_result")

print("✔ evaluation_results")

print("✔ training_loss_values")

print("✔ validation_loss_values")

print("✔ FINAL_MODEL_PATH")

print("✔ generated_text")

print("✔ second_generated_text")


# ============================================================
# SECTION 24 : PART-3 COMPLETION
# ============================================================

print()

print("=" * 70)

print("PART-3 COMPLETED SUCCESSFULLY 🚀")

print("=" * 70)

print()

print("COMPLETED:")

print()

print("✔ LoRA Training Configuration")

print("✔ Hugging Face Trainer")

print("✔ CPU-Friendly Fine-Tuning")

print("✔ Enterprise Dataset Training")

print("✔ Validation")

print("✔ Loss Tracking")

print("✔ Perplexity Calculation")

print("✔ LoRA Adapter Saving")

print("✔ Enterprise Question Testing")

print()

print("READY FOR PART-4")

print()

print("NEXT:")

print("LOAD SAVED LoRA ADAPTER + ENTERPRISE INFERENCE")
# ============================================================
# ENTERPRISE AI — FINE-TUNING A SMALL LANGUAGE MODEL
# PART-4 : FINAL INFERENCE + ENTERPRISE AI APPLICATION
#
# CONTINUES FROM PART-3
#
# No dataset creation
# No retraining
# No repeated LoRA setup
#
# CPU FRIENDLY
# LOW STORAGE
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel


print("Part-4 Libraries Imported Successfully")


# ============================================================
# SECTION 2 : CHECK SAVED MODEL
# ============================================================

print()
print("=" * 70)
print("CHECKING SAVED LoRA ADAPTER")
print("=" * 70)
print()


if os.path.exists(FINAL_MODEL_PATH):

    print("✔ LoRA Adapter Found")

    print(
        "Location:",
        FINAL_MODEL_PATH
    )

else:

    raise FileNotFoundError(
        "LoRA adapter was not found. "
        "Please run Part-3 first."
    )


# ============================================================
# SECTION 3 : CHECK SAVED FILES
# ============================================================

print()
print("=" * 70)
print("SAVED MODEL FILES")
print("=" * 70)
print()


saved_files = os.listdir(
    FINAL_MODEL_PATH
)


for file_name in saved_files:

    print("✔", file_name)


# ============================================================
# SECTION 4 : DEFINE DEVICE
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


print()
print("Inference Device:", device)


# ============================================================
# SECTION 5 : LOAD TOKENIZER
# ============================================================

print()
print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)
print()


inference_tokenizer = AutoTokenizer.from_pretrained(

    FINAL_MODEL_PATH

)


if inference_tokenizer.pad_token is None:

    inference_tokenizer.pad_token = (
        inference_tokenizer.eos_token
    )


print("Tokenizer Loaded Successfully")


# ============================================================
# SECTION 6 : LOAD BASE MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BASE MODEL")
print("=" * 70)
print()


base_model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME

)


base_model.config.pad_token_id = (

    inference_tokenizer.pad_token_id

)


print("Base Model Loaded Successfully")


# ============================================================
# SECTION 7 : LOAD LoRA ADAPTER
# ============================================================

print()
print("=" * 70)
print("LOADING LoRA ADAPTER")
print("=" * 70)
print()


inference_model = PeftModel.from_pretrained(

    base_model,

    FINAL_MODEL_PATH

)


# Move model to CPU/GPU.

inference_model = inference_model.to(

    device

)


# Evaluation mode.

inference_model.eval()


print()
print("LoRA Adapter Loaded Successfully")
print("Model Ready for Inference")


# ============================================================
# SECTION 8 : DISPLAY MODEL INFORMATION
# ============================================================

print()
print("=" * 70)
print("MODEL INFORMATION")
print("=" * 70)
print()


total_parameters = sum(

    parameter.numel()

    for parameter in inference_model.parameters()

)


print(

    "Base Model:",

    MODEL_NAME

)


print(

    "Fine-Tuning:",

    "LoRA / PEFT"

)


print(

    "Total Parameters:",

    f"{total_parameters:,}"

)


print(

    "Device:",

    device

)


# ============================================================
# SECTION 9 : CREATE ENTERPRISE AI FUNCTION
# ============================================================

def ask_enterprise_ai(

    question,

    category="General",

    max_new_tokens=40

):

    """
    Generate an enterprise-style response.

    Parameters
    ----------
    question : str
        Employee's enterprise question.

    category : str
        Enterprise category.

    max_new_tokens : int
        Maximum number of tokens generated.

    Returns
    -------
    str
        Generated enterprise response.
    """


    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    prompt = (

        "### Enterprise Support Task\n\n"

        "Category: "

        + category

        + "\n\n"

        "Question: "

        + question

        + "\n\n"

        "Answer:"

    )


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = inference_tokenizer(

        prompt,

        return_tensors="pt",

        truncation=True,

        max_length=MAX_LENGTH

    )


    # --------------------------------------------------------
    # MOVE INPUT TO DEVICE
    # --------------------------------------------------------

    inputs = {

        key: value.to(device)

        for key, value in inputs.items()

    }


    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    start_time = time.time()


    with torch.no_grad():

        output = inference_model.generate(

            **inputs,

            max_new_tokens=max_new_tokens,

            do_sample=False,

            pad_token_id=(
                inference_tokenizer.pad_token_id
            ),

            eos_token_id=(
                inference_tokenizer.eos_token_id
            )

        )


    end_time = time.time()


    # --------------------------------------------------------
    # DECODE RESPONSE
    # --------------------------------------------------------

    generated_text = inference_tokenizer.decode(

        output[0],

        skip_special_tokens=True

    )


    # --------------------------------------------------------
    # EXTRACT ANSWER
    # --------------------------------------------------------

    if "Answer:" in generated_text:

        answer = generated_text.split(

            "Answer:",

            1

        )[1].strip()

    else:

        answer = generated_text


    # --------------------------------------------------------
    # REMOVE POSSIBLE EXTRA SECTIONS
    # --------------------------------------------------------

    if "###" in answer:

        answer = answer.split(

            "###",

            1

        )[0].strip()


    # --------------------------------------------------------
    # DISPLAY PERFORMANCE
    # --------------------------------------------------------

    inference_time = end_time - start_time


    print()

    print("=" * 70)

    print("ENTERPRISE AI RESPONSE")

    print("=" * 70)

    print()

    print("Category:", category)

    print()

    print("Question:")

    print(question)

    print()

    print("Answer:")

    print(answer)

    print()

    print(

        "Inference Time:",

        round(

            inference_time,

            3

        ),

        "seconds"

    )


    return answer


# ============================================================
# SECTION 10 : TEST CASE 1 — PASSWORD
# ============================================================

print()
print("=" * 70)
print("TEST CASE 1")
print("=" * 70)


response_1 = ask_enterprise_ai(

    question=(
        "How do I reset my corporate password?"
    ),

    category="IT Support"

)


# ============================================================
# SECTION 11 : TEST CASE 2 — VPN
# ============================================================

print()
print("=" * 70)
print("TEST CASE 2")
print("=" * 70)


response_2 = ask_enterprise_ai(

    question=(
        "How can I connect to the corporate VPN?"
    ),

    category="VPN"

)


# ============================================================
# SECTION 12 : TEST CASE 3 — HR
# ============================================================

print()
print("=" * 70)
print("TEST CASE 3")
print("=" * 70)


response_3 = ask_enterprise_ai(

    question=(
        "How do I apply for leave?"
    ),

    category="HR"

)


# ============================================================
# SECTION 13 : TEST CASE 4 — SECURITY
# ============================================================

print()
print("=" * 70)
print("TEST CASE 4")
print("=" * 70)


response_4 = ask_enterprise_ai(

    question=(
        "What should I do if I receive a suspicious email?"
    ),

    category="Security"

)


# ============================================================
# SECTION 14 : TEST CASE 5 — ACCESS MANAGEMENT
# ============================================================

print()
print("=" * 70)
print("TEST CASE 5")
print("=" * 70)


response_5 = ask_enterprise_ai(

    question=(
        "How do I request access to an internal application?"
    ),

    category="Access Management"

)


# ============================================================
# SECTION 15 : CREATE TEST QUERY COLLECTION
# ============================================================

test_queries = [

    {
        "category": "IT Support",
        "question": "My company laptop is very slow."
    },

    {
        "category": "VPN",
        "question": "My VPN keeps disconnecting."
    },

    {
        "category": "HR",
        "question": "Where can I view my payslips?"
    },

    {
        "category": "Security",
        "question": "Can I share my corporate password?"
    },

    {
        "category": "Access Management",
        "question": "My application access was removed."
    }

]


# ============================================================
# SECTION 16 : RUN MULTIPLE ENTERPRISE QUERIES
# ============================================================

print()
print("=" * 70)
print("MULTI-QUERY ENTERPRISE TEST")
print("=" * 70)


all_responses = []


for query in test_queries:


    print()

    print("-" * 70)

    print(

        "Category:",

        query["category"]

    )

    print(

        "Question:",

        query["question"]

    )


    response = ask_enterprise_ai(

        question=query["question"],

        category=query["category"],

        max_new_tokens=35

    )


    all_responses.append(

        {

            "category": query["category"],

            "question": query["question"],

            "response": response

        }

    )


# ============================================================
# SECTION 17 : CREATE RESULTS DATAFRAME
# ============================================================

import pandas as pd


results_df = pd.DataFrame(

    all_responses

)


print()
print("=" * 70)
print("ENTERPRISE AI TEST RESULTS")
print("=" * 70)
print()


display(results_df)


# ============================================================
# SECTION 18 : CHECK RESPONSE COUNT
# ============================================================

print()

print(

    "Total Test Queries:",

    len(results_df)

)


print(

    "Generated Responses:",

    results_df["response"].notnull().sum()

)


# ============================================================
# SECTION 19 : SIMPLE RESPONSE QUALITY CHECK
# ============================================================

print()
print("=" * 70)
print("RESPONSE QUALITY CHECK")
print("=" * 70)
print()


results_df["response_length"] = (

    results_df["response"]

    .astype(str)

    .apply(

        lambda x: len(x.split())

    )

)


display(

    results_df[

        [

            "category",

            "question",

            "response_length"

        ]

    ]

)


# ============================================================
# SECTION 20 : SAVE TEST RESULTS
# ============================================================

# This is a very small CSV and requires negligible storage.

RESULTS_FILE = (

    "enterprise_ai_results.csv"

)


results_df.to_csv(

    RESULTS_FILE,

    index=False

)


print()

print(

    "Test Results Saved:",

    RESULTS_FILE

)


# ============================================================
# SECTION 21 : FINAL MODEL SIZE CHECK
# ============================================================

print()
print("=" * 70)
print("LoRA ADAPTER STORAGE CHECK")
print("=" * 70)
print()


total_size_bytes = 0


for root, directories, files in os.walk(

    FINAL_MODEL_PATH

):

    for file in files:

        file_path = os.path.join(

            root,

            file

        )

        total_size_bytes += os.path.getsize(

            file_path

        )


total_size_mb = (

    total_size_bytes

    /

    (

        1024 * 1024

    )

)


print(

    "LoRA Adapter Size:",

    round(

        total_size_mb,

        2

    ),

    "MB"

)


# ============================================================
# SECTION 22 : FINAL PROJECT ARCHITECTURE
# ============================================================

print()
print("=" * 70)
print("FINAL ENTERPRISE AI ARCHITECTURE")
print("=" * 70)
print()


print(
    """
    Employee
       |
       v
Enterprise Question
       |
       v
Category + Question
       |
       v
Enterprise AI
       |
       +----------------------+
       |                      |
       v                      v
Base GPT Model          LoRA Adapter
       |                      |
       +----------+-----------+
                  |
                  v
          Generated Response
                  |
                  v
              Employee
    """
)


# ============================================================
# SECTION 23 : COMPLETE PROJECT SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMPLETE PROJECT SUMMARY")
print("=" * 70)
print()


print("PROJECT : Enterprise AI Fine-Tuning")


print()
print("1. Dataset")
print("   ✔ Custom enterprise support dataset")


print()
print("2. Preprocessing")
print("   ✔ Text cleaning")
print("   ✔ Train / validation split")
print("   ✔ Instruction formatting")


print()
print("3. Base Model")
print("   ✔ distilgpt2")


print()
print("4. Fine-Tuning")
print("   ✔ LoRA")
print("   ✔ PEFT")
print("   ✔ CPU-friendly configuration")


print()
print("5. Training")
print("   ✔ Hugging Face Trainer")
print("   ✔ Validation")
print("   ✔ Loss tracking")
print("   ✔ Perplexity")


print()
print("6. Model Saving")
print("   ✔ LoRA adapter saved")


print()
print("7. Inference")
print("   ✔ Enterprise question answering")
print("   ✔ Multiple test queries")


print()
print("8. Evaluation")
print("   ✔ Response generation")
print("   ✔ Response length analysis")


# ============================================================
# SECTION 24 : FINAL COMPLETION
# ============================================================

print()
print("=" * 70)
print("ENTERPRISE AI PROJECT COMPLETED 🚀")
print("=" * 70)
print()


print(
    """
    PART-1 ✔ Dataset + Preprocessing

    PART-2 ✔ Small GPT Model + LoRA

    PART-3 ✔ Fine-Tuning + Validation

    PART-4 ✔ Inference + Evaluation
    """
)


print(
    "Your complete Enterprise AI fine-tuning pipeline is ready!"
)
