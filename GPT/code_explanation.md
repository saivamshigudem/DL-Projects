# ============================================================
#  DATA PREPARATION & EDA
# PROJECT : TEXT GENERATION USING GPT ARCHITECTURE
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# ============================================================
# SECTION 2 : CREATE PREDEFINED TEXT DATASET
# ============================================================

# This is a small predefined corpus created specifically
# for learning and training a mini GPT model on CPU.
#
# The dataset focuses on:
# Artificial Intelligence
# Machine Learning
# Deep Learning
# Python
# Data Science
# Neural Networks
# Transformers
# GPT

raw_text = """

Artificial intelligence is transforming the world.
Artificial intelligence helps machines perform intelligent tasks.
Machine learning is a branch of artificial intelligence.
Machine learning helps computers learn patterns from data.
Deep learning is a powerful technique in artificial intelligence.
Deep learning uses neural networks with multiple layers.

Neural networks are inspired by the human brain.
Neural networks learn complex patterns from training data.
A neural network contains multiple connected layers.
The input layer receives information from the dataset.
Hidden layers learn important patterns from the input.
The output layer produces the final prediction.

Data is an important part of machine learning.
High quality data can improve machine learning models.
Data science combines programming statistics and domain knowledge.
Python is widely used in data science and artificial intelligence.
Python provides many useful libraries for machine learning.

PyTorch is a popular framework for deep learning.
PyTorch allows developers to build neural networks.
TensorFlow is another popular deep learning framework.
Machine learning models learn from examples.
Training data helps models understand patterns.

A model makes predictions based on learned patterns.
Model training requires data computation and optimization.
Optimization helps reduce the prediction error.
Loss functions measure the error made by a model.
Gradient descent helps reduce the loss function.

Natural language processing helps computers understand language.
Natural language processing is also called NLP.
NLP allows machines to process human language.
Text classification is an important NLP task.
Sentiment analysis identifies positive or negative opinions.

Transformers changed the field of natural language processing.
Transformers use attention mechanisms to understand relationships.
Attention helps models focus on important information.
Self attention allows tokens to interact with other tokens.
Multi head attention allows models to learn different relationships.

GPT stands for Generative Pretrained Transformer.
GPT models generate text by predicting the next token.
GPT uses a decoder based transformer architecture.
GPT models use causal self attention.
Causal attention prevents the model from seeing future tokens.

Text generation predicts one token at a time.
The generated token becomes part of the next input.
This process continues until the required text is generated.
Language models learn patterns from large amounts of text.

A good model requires good training data.
More training data can improve model performance.
Better architecture can improve model performance.
Hyperparameters control the training process.
The learning rate controls how quickly a model learns.

Artificial intelligence will continue to evolve.
Machine learning will become more powerful.
Deep learning will solve complex problems.
Transformers will continue to improve language models.
GPT models demonstrate the power of generative artificial intelligence.

Python makes artificial intelligence development easier.
Developers use Python to build machine learning applications.
Data scientists use Python to analyze data.
Engineers use deep learning to build intelligent systems.
Researchers use transformers to solve language problems.

Artificial intelligence combines data algorithms and computation.
Machine learning discovers patterns in data.
Deep learning learns hierarchical representations.
Neural networks process information through connected layers.
Transformers process sequences using attention mechanisms.

Learning requires practice and experimentation.
Projects help developers understand machine learning concepts.
Experiments help researchers improve model performance.
Evaluation helps measure the quality of predictions.
Continuous learning helps engineers build better systems.

"""


# ============================================================
# SECTION 3 : DISPLAY ORIGINAL TEXT
# ============================================================

print("FIRST 1000 CHARACTERS OF THE DATASET")

print()

print(raw_text[:1000])


# ============================================================
# SECTION 4 : CLEAN THE TEXT
# ============================================================

# Convert all text to lowercase so that words such as:
#
# Artificial
# artificial
# ARTIFICIAL
#
# are treated as the same word.

cleaned_text = raw_text.lower()


# Remove unnecessary extra spaces.

cleaned_text = re.sub(

    r"\s+",

    " ",

    cleaned_text

)


# Remove leading and trailing spaces.

cleaned_text = cleaned_text.strip()


print("TEXT CLEANING COMPLETED SUCCESSFULLY")


# ============================================================
# SECTION 5 : CREATE SENTENCE DATASET
# ============================================================

# Split the original text into sentences.
#
# The GPT model will later use the complete corpus,
# but sentence-level analysis is useful for EDA.

sentences = [

    sentence.strip()

    for sentence in raw_text.split(".")

    if sentence.strip()

]


# Create a Pandas DataFrame.

text_df = pd.DataFrame(

    sentences,

    columns=["sentence"]

)


# ============================================================
# SECTION 6 : DISPLAY DATASET
# ============================================================

print()

print("NUMBER OF SENTENCES :", len(text_df))

print()

display(text_df.head(10))


# ============================================================
# SECTION 7 : CHARACTER ANALYSIS
# ============================================================

total_characters = len(cleaned_text)

print("TOTAL CHARACTERS")

print(total_characters)


# ============================================================
# SECTION 8 : WORD ANALYSIS
# ============================================================

# Split the cleaned text into individual words.

words = cleaned_text.split()


total_words = len(words)


print()

print("TOTAL WORDS")

print(total_words)


# ============================================================
# SECTION 9 : VOCABULARY ANALYSIS
# ============================================================

# Find all unique words.

unique_words = set(words)


vocabulary_size = len(unique_words)


print()

print("VOCABULARY SIZE")

print(vocabulary_size)


# ============================================================
# SECTION 10 : AVERAGE SENTENCE LENGTH
# ============================================================

# Calculate the number of words in every sentence.

text_df["word_count"] = text_df["sentence"].apply(

    lambda sentence: len(sentence.split())

)


average_sentence_length = text_df["word_count"].mean()


print()

print("AVERAGE SENTENCE LENGTH")

print(

    round(

        average_sentence_length,

        2

    ),

    "words"

)


# ============================================================
# SECTION 11 : SENTENCE LENGTH STATISTICS
# ============================================================

print()

print("SENTENCE LENGTH STATISTICS")

print()

print(

    text_df["word_count"].describe()

)


# ============================================================
# SECTION 12 : SENTENCE LENGTH DISTRIBUTION
# ============================================================

plt.figure(

    figsize=(8, 5)

)


plt.hist(

    text_df["word_count"],

    bins=10

)


plt.title(

    "Sentence Length Distribution"

)


plt.xlabel(

    "Number of Words"

)


plt.ylabel(

    "Number of Sentences"

)


plt.grid(

    True

)


plt.show()


# ============================================================
# SECTION 13 : WORD FREQUENCY ANALYSIS
# ============================================================

word_frequency = Counter(

    words

)


print()

print("TOP 20 MOST COMMON WORDS")

print()


for word, count in word_frequency.most_common(20):

    print(

        word,

        ":",

        count

    )


# ============================================================
# SECTION 14 : WORD FREQUENCY DATAFRAME
# ============================================================

frequency_df = pd.DataFrame(

    word_frequency.items(),

    columns=[

        "word",

        "frequency"

    ]

)


frequency_df = frequency_df.sort_values(

    by="frequency",

    ascending=False

)


print()

display(

    frequency_df.head(20)

)


# ============================================================
# SECTION 15 : VISUALIZE TOP WORDS
# ============================================================

top_words = frequency_df.head(15)


plt.figure(

    figsize=(10, 6)

)


plt.bar(

    top_words["word"],

    top_words["frequency"]

)


plt.title(

    "Top 15 Most Frequent Words"

)


plt.xlabel(

    "Words"

)


plt.ylabel(

    "Frequency"

)


plt.xticks(

    rotation=45

)


plt.grid(

    True,

    axis="y"

)


plt.tight_layout()

plt.show()


# ============================================================
# SECTION 16 : LONGEST SENTENCE
# ============================================================

longest_sentence_index = text_df[

    "word_count"

].idxmax()


longest_sentence = text_df.loc[

    longest_sentence_index,

    "sentence"

]


print()

print("LONGEST SENTENCE")

print()

print(

    longest_sentence

)


print()

print(

    "WORD COUNT :",

    text_df.loc[

        longest_sentence_index,

        "word_count"

    ]

)


# ============================================================
# SECTION 17 : SHORTEST SENTENCE
# ============================================================

shortest_sentence_index = text_df[

    "word_count"

].idxmin()


shortest_sentence = text_df.loc[

    shortest_sentence_index,

    "sentence"

]


print()

print("SHORTEST SENTENCE")

print()

print(

    shortest_sentence

)


print()

print(

    "WORD COUNT :",

    text_df.loc[

        shortest_sentence_index,

        "word_count"

    ]

)


# ============================================================
# SECTION 18 : PREPARE FINAL TEXT CORPUS
# ============================================================

# This final corpus will be used in Part-2
# for vocabulary creation and tokenization.

text_corpus = cleaned_text


print()

print("FINAL TEXT CORPUS READY")


# ============================================================
# SECTION 19 : DATASET SUMMARY
# ============================================================

print()

print()

print("PROJECT")

print("Text Generation Using GPT Architecture")

print()

print("DATASET TYPE")

print("Predefined AI and Machine Learning Text Corpus")

print()

print("TOTAL SENTENCES")

print(len(text_df))

print()

print("TOTAL CHARACTERS")

print(total_characters)

print()

print("TOTAL WORDS")

print(total_words)

print()

print("VOCABULARY SIZE")

print(vocabulary_size)

print()

print("AVERAGE SENTENCE LENGTH")

print(

    round(

        average_sentence_length,

        2

    )

)

print()

print("CPU FRIENDLY")

print("Yes")

print()


print("Tokenization and GPT Input Preparation")
# ============================================================
#  TOKENIZATION & GPT INPUT PREPARATION
# PROJECT : TEXT GENERATION USING GPT ARCHITECTURE
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import torch

from torch.utils.data import Dataset, DataLoader


# ============================================================
# SECTION 2 : CREATE SPECIAL TOKENS
# ============================================================

# Special tokens help us handle unknown words
# and padding when required.

PAD_TOKEN = "<PAD>"

UNK_TOKEN = "<UNK>"


# ============================================================
# SECTION 3 : CREATE VOCABULARY
# ============================================================

# Get all unique words from the corpus.

unique_words = sorted(

    list(

        set(

            words

        )

    )

)


# Add special tokens at the beginning.

vocabulary = [

    PAD_TOKEN,

    UNK_TOKEN

] + unique_words


# Calculate final vocabulary size.

vocab_size = len(

    vocabulary

)


print("VOCABULARY CREATED SUCCESSFULLY")

print()

print("VOCABULARY SIZE :", vocab_size)


# ============================================================
# SECTION 4 : CREATE WORD TO INDEX MAPPING
# ============================================================

# Example:
#
# "artificial" -> 10
# "intelligence" -> 25

word_to_index = {

    word: index

    for index, word in enumerate(

        vocabulary

    )

}


print()

print("WORD TO INDEX MAPPING CREATED")


# ============================================================
# SECTION 5 : CREATE INDEX TO WORD MAPPING
# ============================================================

# Example:
#
# 10 -> "artificial"
# 25 -> "intelligence"

index_to_word = {

    index: word

    for word, index in word_to_index.items()

}


print()

print("INDEX TO WORD MAPPING CREATED")


# ============================================================
# SECTION 6 : DISPLAY VOCABULARY SAMPLE
# ============================================================

print()

print("FIRST 20 VOCABULARY WORDS")

print()

print(

    vocabulary[:20]

)


# ============================================================
# SECTION 7 : TEST WORD MAPPING
# ============================================================

sample_word = "artificial"


print()

print("SAMPLE WORD :", sample_word)

print(

    "TOKEN ID :",

    word_to_index[sample_word]

)


print(

    "REVERSE MAPPING :",

    index_to_word[

        word_to_index[sample_word]

    ]

)


# ============================================================
# SECTION 8 : CONVERT TEXT INTO TOKEN IDs
# ============================================================

# Convert every word in the corpus
# into its corresponding numerical ID.

token_ids = [

    word_to_index.get(

        word,

        word_to_index[UNK_TOKEN]

    )

    for word in words

]


print()

print("TOTAL TOKEN IDs :", len(token_ids))


# ============================================================
# SECTION 9 : DISPLAY TOKENIZATION EXAMPLE
# ============================================================

print()

print("ORIGINAL WORDS")

print(

    words[:20]

)


print()

print("TOKEN IDs")

print(

    token_ids[:20]

)


# ============================================================
# SECTION 10 : DEFINE SEQUENCE LENGTH
# ============================================================

# CONTEXT_LENGTH determines how many previous tokens
# the GPT model can look at.
#
# 32 is small and CPU-friendly.

CONTEXT_LENGTH = 32


print()

print(

    "CONTEXT LENGTH :",

    CONTEXT_LENGTH

)


# ============================================================
# SECTION 11 : CREATE INPUT SEQUENCES AND TARGETS
# ============================================================

# Example:
#
# Input:
# artificial intelligence is transforming
#
# Target:
# the
#
# The model learns to predict the next token.

input_sequences = []

target_tokens = []


for i in range(

    len(token_ids) - CONTEXT_LENGTH

):


    # Get CONTEXT_LENGTH tokens as input.

    input_sequence = token_ids[

        i : i + CONTEXT_LENGTH

    ]


    # Get the next token as target.

    target_token = token_ids[

        i + CONTEXT_LENGTH

    ]


    input_sequences.append(

        input_sequence

    )


    target_tokens.append(

        target_token

    )


print()

print(

    "TOTAL TRAINING SEQUENCES :",

    len(input_sequences)

)


# ============================================================
# SECTION 12 : CONVERT TO PYTORCH TENSORS
# ============================================================

X = torch.tensor(

    input_sequences,

    dtype=torch.long

)


y = torch.tensor(

    target_tokens,

    dtype=torch.long

)


print()

print("INPUT SHAPE :", X.shape)

print("TARGET SHAPE :", y.shape)


# ============================================================
# SECTION 13 : CREATE CUSTOM PYTORCH DATASET
# ============================================================

class GPTDataset(

    Dataset

):


    def __init__(

        self,

        inputs,

        targets

    ):


        self.inputs = inputs

        self.targets = targets


    def __len__(

        self

    ):


        return len(

            self.inputs

        )


    def __getitem__(

        self,

        index

    ):


        return (

            self.inputs[index],

            self.targets[index]

        )


# ============================================================
# SECTION 14 : CREATE DATASET OBJECT
# ============================================================

train_dataset = GPTDataset(

    X,

    y

)


print()

print(

    "DATASET SIZE :",

    len(train_dataset)

)


# ============================================================
# SECTION 15 : CREATE DATALOADER
# ============================================================

# Small batch size keeps training manageable on CPU.

BATCH_SIZE = 16


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


print()

print(

    "NUMBER OF BATCHES :",

    len(train_loader)

)


# ============================================================
# SECTION 16 : INSPECT ONE TRAINING SAMPLE
# ============================================================

sample_input = X[0]

sample_target = y[0]


# Convert token IDs back to words.

input_words = [

    index_to_word[

        token.item()

    ]

    for token in sample_input

]


target_word = index_to_word[

    sample_target.item()

]


print()

print("=" * 60)

print("INPUT SEQUENCE")

print("=" * 60)

print(

    " ".join(

        input_words

    )

)


print()

print("=" * 60)

print("TARGET WORD")

print("=" * 60)

print(

    target_word

)


# ============================================================
# SECTION 17 : INSPECT ONE BATCH
# ============================================================

batch_inputs, batch_targets = next(

    iter(train_loader)

)


print()

print("BATCH INPUT SHAPE")

print(

    batch_inputs.shape

)


print()

print("BATCH TARGET SHAPE")

print(

    batch_targets.shape

)


# ============================================================
# SECTION 18 : DECODE FIRST INPUT FROM BATCH
# ============================================================

first_sequence = batch_inputs[0]


decoded_sequence = [

    index_to_word[

        token.item()

    ]

    for token in first_sequence

]


target = index_to_word[

    batch_targets[0].item()

]


print()

print("FIRST BATCH INPUT")

print()

print(

    " ".join(

        decoded_sequence

    )

)


print()

print("EXPECTED NEXT WORD")

print()

print(

    target

)


# ============================================================
# SECTION 19 : DEVICE CONFIGURATION
# ============================================================

# Automatically use GPU if available.
# Otherwise, continue with CPU.

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print()

print(

    "RUNNING ON :",

    device

)


# ============================================================
# SECTION 20 :  SUMMARY
# ============================================================

print()

print()

print("VOCABULARY SIZE :", vocab_size)

print("TOTAL TOKENS :", len(token_ids))

print("CONTEXT LENGTH :", CONTEXT_LENGTH)

print("TOTAL TRAINING SEQUENCES :", len(train_dataset))

print("BATCH SIZE :", BATCH_SIZE)

print("NUMBER OF BATCHES :", len(train_loader))

print()

print("CREATED VARIABLES")

print()

print("✔ vocabulary")

print("✔ word_to_index")

print("✔ index_to_word")

print("✔ vocab_size")

print("✔ token_ids")

print("✔ X")

print("✔ y")

print("✔ train_dataset")

print("✔ train_loader")

print("✔ CONTEXT_LENGTH")

print()

print("GPT ARCHITECTURE & MODEL TRAINING")

# ============================================================
#  BUILD GPT ARCHITECTURE & TRAINING
# PROJECT : TEXT GENERATION USING GPT ARCHITECTURE
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# SECTION 2 : SET MODEL HYPERPARAMETERS
# ============================================================

# Keep the model intentionally small because
# the project is designed for CPU training.

EMBEDDING_DIM = 64

NUM_HEADS = 4

NUM_LAYERS = 2

DROPOUT = 0.1

LEARNING_RATE = 0.001

EPOCHS = 30


print("MODEL CONFIGURATION")

print("Vocabulary Size :", vocab_size)

print("Embedding Dimension :", EMBEDDING_DIM)

print("Number of Attention Heads :", NUM_HEADS)

print("Number of Transformer Layers :", NUM_LAYERS)

print("Context Length :", CONTEXT_LENGTH)

print("Epochs :", EPOCHS)


# ============================================================
# SECTION 3 : CREATE CAUSAL SELF-ATTENTION CLASS
# ============================================================

# GPT uses causal attention.
#
# A token should only attend to:
#
# Current token
# Previous tokens
#
# It should NOT look at future tokens.

class CausalSelfAttention(nn.Module):


    def __init__(

        self,

        embedding_dim,

        num_heads,

        context_length,

        dropout

    ):


        super().__init__()


        # Check whether embedding dimension
        # can be divided equally among attention heads.

        assert embedding_dim % num_heads == 0


        self.num_heads = num_heads

        self.embedding_dim = embedding_dim

        self.head_dim = (

            embedding_dim // num_heads

        )


        # Linear layers for:
        #
        # Query
        # Key
        # Value

        self.query = nn.Linear(

            embedding_dim,

            embedding_dim

        )


        self.key = nn.Linear(

            embedding_dim,

            embedding_dim

        )


        self.value = nn.Linear(

            embedding_dim,

            embedding_dim

        )


        # Output projection.

        self.output = nn.Linear(

            embedding_dim,

            embedding_dim

        )


        # Dropout for regularization.

        self.dropout = nn.Dropout(

            dropout

        )


        # Create lower triangular causal mask.
        #
        # Example:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1

        mask = torch.tril(

            torch.ones(

                context_length,

                context_length

            )

        )


        # Register mask as part of the model.

        self.register_buffer(

            "mask",

            mask

        )


    def forward(

        self,

        x

    ):


        # Get batch size and sequence length.

        batch_size, sequence_length, embedding_dim = x.shape


        # Create Query, Key and Value.

        Q = self.query(

            x

        )


        K = self.key(

            x

        )


        V = self.value(

            x

        )


        # Reshape tensors for multi-head attention.
        #
        # From:
        #
        # batch, sequence, embedding
        #
        # To:
        #
        # batch, heads, sequence, head_dimension

        Q = Q.view(

            batch_size,

            sequence_length,

            self.num_heads,

            self.head_dim

        ).transpose(

            1,

            2

        )


        K = K.view(

            batch_size,

            sequence_length,

            self.num_heads,

            self.head_dim

        ).transpose(

            1,

            2

        )


        V = V.view(

            batch_size,

            sequence_length,

            self.num_heads,

            self.head_dim

        ).transpose(

            1,

            2

        )


        # Calculate attention scores.
        #
        # Attention =
        #
        # Q × K transpose

        attention_scores = torch.matmul(

            Q,

            K.transpose(

                -2,

                -1

            )

        )


        # Scale attention scores.

        attention_scores = attention_scores / (

            self.head_dim ** 0.5

        )


        # Apply causal mask.
        #
        # Future tokens receive negative infinity
        # so the softmax probability becomes zero.

        attention_scores = attention_scores.masked_fill(

            self.mask[

                :sequence_length,

                :sequence_length

            ] == 0,

            float("-inf")

        )


        # Convert attention scores
        # into probabilities.

        attention_weights = F.softmax(

            attention_scores,

            dim=-1

        )


        # Apply dropout.

        attention_weights = self.dropout(

            attention_weights

        )


        # Multiply attention weights
        # with Value.

        attention_output = torch.matmul(

            attention_weights,

            V

        )


        # Rearrange dimensions back.

        attention_output = attention_output.transpose(

            1,

            2

        ).contiguous()


        # Combine all attention heads.

        attention_output = attention_output.view(

            batch_size,

            sequence_length,

            embedding_dim

        )


        # Final linear projection.

        output = self.output(

            attention_output

        )


        return output


# ============================================================
# SECTION 4 : CREATE FEED FORWARD NETWORK
# ============================================================

# Every Transformer block contains
# a feed forward neural network.

class FeedForward(nn.Module):


    def __init__(

        self,

        embedding_dim,

        dropout

    ):


        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(

                embedding_dim,

                embedding_dim * 4

            ),

            nn.ReLU(),

            nn.Linear(

                embedding_dim * 4,

                embedding_dim

            ),

            nn.Dropout(

                dropout

            )

        )


    def forward(

        self,

        x

    ):


        return self.network(

            x

        )


# ============================================================
# SECTION 5 : CREATE TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):


    def __init__(

        self,

        embedding_dim,

        num_heads,

        context_length,

        dropout

    ):


        super().__init__()


        # First Layer Normalization.

        self.layer_norm_1 = nn.LayerNorm(

            embedding_dim

        )


        # Causal Self Attention.

        self.attention = CausalSelfAttention(

            embedding_dim,

            num_heads,

            context_length,

            dropout

        )


        # Second Layer Normalization.

        self.layer_norm_2 = nn.LayerNorm(

            embedding_dim

        )


        # Feed Forward Network.

        self.feed_forward = FeedForward(

            embedding_dim,

            dropout

        )


    def forward(

        self,

        x

    ):


        # Residual connection around attention.

        x = x + self.attention(

            self.layer_norm_1(

                x

            )

        )


        # Residual connection around feed forward network.

        x = x + self.feed_forward(

            self.layer_norm_2(

                x

            )

        )


        return x


# ============================================================
# SECTION 6 : CREATE MINI GPT MODEL
# ============================================================

class MiniGPT(nn.Module):


    def __init__(

        self,

        vocab_size,

        embedding_dim,

        num_heads,

        num_layers,

        context_length,

        dropout

    ):


        super().__init__()


        # Convert token IDs into dense vectors.

        self.token_embedding = nn.Embedding(

            vocab_size,

            embedding_dim

        )


        # Learn positional information.

        self.position_embedding = nn.Embedding(

            context_length,

            embedding_dim

        )


        # Dropout after embeddings.

        self.embedding_dropout = nn.Dropout(

            dropout

        )


        # Create multiple Transformer blocks.

        self.transformer_blocks = nn.Sequential(

            *[

                TransformerBlock(

                    embedding_dim,

                    num_heads,

                    context_length,

                    dropout

                )

                for _ in range(

                    num_layers

                )

            ]

        )


        # Final layer normalization.

        self.final_layer_norm = nn.LayerNorm(

            embedding_dim

        )


        # Language modeling head.
        #
        # Converts embeddings into vocabulary scores.

        self.language_model_head = nn.Linear(

            embedding_dim,

            vocab_size

        )


    def forward(

        self,

        input_ids

    ):


        batch_size, sequence_length = input_ids.shape


        # Create position IDs.
        #
        # Example:
        #
        # [0, 1, 2, 3, ...]

        positions = torch.arange(

            sequence_length,

            device=input_ids.device

        )


        # Token embeddings.

        token_embeddings = self.token_embedding(

            input_ids

        )


        # Position embeddings.

        position_embeddings = self.position_embedding(

            positions

        )


        # Combine token and position embeddings.

        x = (

            token_embeddings +

            position_embeddings

        )


        # Apply dropout.

        x = self.embedding_dropout(

            x

        )


        # Pass through Transformer blocks.

        x = self.transformer_blocks(

            x

        )


        # Final normalization.

        x = self.final_layer_norm(

            x

        )


        # Generate vocabulary logits.

        logits = self.language_model_head(

            x

        )


        return logits


# ============================================================
# SECTION 7 : CREATE GPT MODEL
# ============================================================

model = MiniGPT(

    vocab_size=vocab_size,

    embedding_dim=EMBEDDING_DIM,

    num_heads=NUM_HEADS,

    num_layers=NUM_LAYERS,

    context_length=CONTEXT_LENGTH,

    dropout=DROPOUT

)


# Move model to CPU or GPU.

model = model.to(

    device

)


print()

print("MINI GPT MODEL CREATED SUCCESSFULLY")

print()

print(model)


# ============================================================
# SECTION 8 : COUNT MODEL PARAMETERS
# ============================================================

total_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

)


print()

print(

    "TOTAL TRAINABLE PARAMETERS :",

    total_parameters

)


# ============================================================
# SECTION 9 : DEFINE LOSS FUNCTION
# ============================================================

# CrossEntropyLoss is used for
# multi-class next-token prediction.

criterion = nn.CrossEntropyLoss()


# ============================================================
# SECTION 10 : DEFINE OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


# ============================================================
# SECTION 11 : TRAINING HISTORY
# ============================================================

training_losses = []


# ============================================================
# SECTION 12 : TRAIN THE MINI GPT MODEL
# ============================================================

print()

print("=" * 60)

print("STARTING MINI GPT TRAINING")

print("=" * 60)


for epoch in range(

    EPOCHS

):


    # Enable training mode.

    model.train()


    total_loss = 0


    # Loop through all training batches.

    for batch_inputs, batch_targets in train_loader:


        # Move data to device.

        batch_inputs = batch_inputs.to(

            device

        )


        batch_targets = batch_targets.to(

            device

        )


        # Clear previous gradients.

        optimizer.zero_grad()


        # Forward pass.
        #
        # Output shape:
        #
        # batch_size
        # × context_length
        # × vocab_size

        outputs = model(

            batch_inputs

        )


        # We only need the prediction
        # from the LAST token position
        # because the target is the
        # next word after the context.

        last_token_logits = outputs[

            :,

            -1,

            :

        ]


        # Calculate prediction loss.

        loss = criterion(

            last_token_logits,

            batch_targets

        )


        # Backpropagation.

        loss.backward()


        # Update model parameters.

        optimizer.step()


        # Add batch loss.

        total_loss += loss.item()


    # Calculate average epoch loss.

    average_loss = (

        total_loss /

        len(train_loader)

    )


    training_losses.append(

        average_loss

    )


    # Display training progress.

    print(

        f"Epoch [{epoch + 1}/{EPOCHS}] "

        f"Loss: {average_loss:.4f}"

    )


# ============================================================
# SECTION 13 : TRAINING COMPLETED
# ============================================================

print()

print("=" * 60)

print("TRAINING COMPLETED SUCCESSFULLY")

print("=" * 60)


# ============================================================
# SECTION 14 : VISUALIZE TRAINING LOSS
# ============================================================

plt.figure(

    figsize=(8, 5)

)


plt.plot(

    range(

        1,

        EPOCHS + 1

    ),

    training_losses,

    marker="o"

)


plt.title(

    "Mini GPT Training Loss"

)


plt.xlabel(

    "Epoch"

)


plt.ylabel(

    "Loss"

)


plt.grid(

    True

)


plt.show()


# ============================================================
# SECTION 15 : SAVE TRAINED MODEL
# ============================================================

MODEL_PATH = "mini_gpt_model.pth"


torch.save(

    model.state_dict(),

    MODEL_PATH

)


print()

print(

    "MODEL SAVED SUCCESSFULLY AT :",

    MODEL_PATH

)


# ============================================================
# SECTION 16 : VERIFY MODEL OUTPUT
# ============================================================

# Get one batch from the DataLoader.

sample_inputs, sample_targets = next(

    iter(train_loader)

)


# Move input to device.

sample_inputs = sample_inputs.to(

    device

)


# Switch to evaluation mode.

model.eval()


with torch.no_grad():


    sample_outputs = model(

        sample_inputs

    )


print()

print("MODEL OUTPUT SHAPE")

print(

    sample_outputs.shape

)


print()

print("EXPECTED OUTPUT SHAPE")

print(

    f"(Batch Size, {CONTEXT_LENGTH}, {vocab_size})"

)


# ============================================================
# SECTION 17 :  SUMMARY
# ============================================================


print("MODEL COMPONENTS CREATED")

print()

print("✔ Token Embeddings")

print("✔ Positional Embeddings")

print("✔ Causal Self-Attention")

print("✔ Multi-Head Attention")

print("✔ Feed Forward Network")

print("✔ Layer Normalization")

print("✔ Residual Connections")

print("✔ Transformer Blocks")

print("✔ Language Model Head")

print()

print("MODEL CONFIGURATION")

print()

print("Embedding Dimension :", EMBEDDING_DIM)

print("Attention Heads :", NUM_HEADS)

print("Transformer Layers :", NUM_LAYERS)

print("Context Length :", CONTEXT_LENGTH)

print("Vocabulary Size :", vocab_size)

print("Epochs :", EPOCHS)

print()

print("FINAL TRAINING LOSS :")

print(

    round(

        training_losses[-1],

        4

    )

)

print()

print("MODEL SAVED AS :")

print(

    MODEL_PATH

)

print()


print("TEXT GENERATION USING THE TRAINED GPT MODEL")

# ============================================================
#  TEXT GENERATION USING MINI GPT
# PROJECT : TEXT GENERATION USING GPT ARCHITECTURE
# FRAMEWORK : PYTORCH
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import torch
import json


# ============================================================
# SECTION 2 : SET MODEL TO EVALUATION MODE
# ============================================================

# Disable dropout and prepare the model
# for text generation.

model.eval()

print("MODEL IS READY FOR TEXT GENERATION")


# ============================================================
# SECTION 3 : GET SPECIAL TOKEN IDS
# ============================================================

PAD_TOKEN_ID = word_to_index["<PAD>"]

UNK_TOKEN_ID = word_to_index["<UNK>"]


print()

print("PAD TOKEN ID :", PAD_TOKEN_ID)

print("UNK TOKEN ID :", UNK_TOKEN_ID)


# ============================================================
# SECTION 4 : CREATE TEXT GENERATION FUNCTION
# ============================================================

def generate_text(
    
    prompt,
    
    max_new_tokens=20,
    
    temperature=1.0,
    
    top_k=None,
    
    deterministic=False
    
):


    # --------------------------------------------------------
    # STEP 1 : CLEAN THE PROMPT
    # --------------------------------------------------------

    prompt = prompt.lower().strip()


    # --------------------------------------------------------
    # STEP 2 : CONVERT PROMPT INTO WORDS
    # --------------------------------------------------------

    prompt_words = prompt.split()


    # --------------------------------------------------------
    # STEP 3 : CONVERT WORDS INTO TOKEN IDS
    # --------------------------------------------------------

    generated_token_ids = [

        word_to_index.get(
            
            word,
            
            UNK_TOKEN_ID
            
        )

        for word in prompt_words

    ]


    # --------------------------------------------------------
    # STEP 4 : GENERATE TOKENS ONE BY ONE
    # --------------------------------------------------------

    for _ in range(max_new_tokens):


        # ----------------------------------------------------
        # GET THE MOST RECENT CONTEXT
        # ----------------------------------------------------

        context = generated_token_ids[-CONTEXT_LENGTH:]


        # ----------------------------------------------------
        # PAD SHORT INPUTS
        # ----------------------------------------------------

        if len(context) < CONTEXT_LENGTH:


            padding_length = (

                CONTEXT_LENGTH - len(context)

            )


            context = (

                [PAD_TOKEN_ID] * padding_length

                + context

            )


        # ----------------------------------------------------
        # CONVERT CONTEXT INTO PYTORCH TENSOR
        # ----------------------------------------------------

        input_tensor = torch.tensor(

            [context],

            dtype=torch.long

        ).to(device)


        # ----------------------------------------------------
        # PREDICTION MODE
        # ----------------------------------------------------

        with torch.no_grad():


            # Get model output.

            output = model(

                input_tensor

            )


            # Get prediction from the last position.

            logits = output[

                0,

                -1,

                :

            ]


        # ----------------------------------------------------
        # APPLY TEMPERATURE
        # ----------------------------------------------------

        # Lower temperature:
        # More predictable output.
        #
        # Higher temperature:
        # More random output.

        temperature = max(

            temperature,

            0.1

        )


        logits = logits / temperature


        # ----------------------------------------------------
        # DETERMINISTIC GENERATION
        # ----------------------------------------------------

        if deterministic:


            next_token_id = torch.argmax(

                logits

            ).item()


        # ----------------------------------------------------
        # TOP-K SAMPLING
        # ----------------------------------------------------

        else:


            if top_k is not None:


                # Make sure top_k does not exceed vocabulary.

                top_k = min(

                    top_k,

                    vocab_size

                )


                # Get top-k values and indices.

                top_values, top_indices = torch.topk(

                    logits,

                    top_k

                )


                # Convert top-k logits to probabilities.

                probabilities = torch.softmax(

                    top_values,

                    dim=-1

                )


                # Randomly select one token.

                selected_index = torch.multinomial(

                    probabilities,

                    num_samples=1

                ).item()


                # Convert selected index
                # back to original vocabulary ID.

                next_token_id = top_indices[

                    selected_index

                ].item()


            # ------------------------------------------------
            # NORMAL SAMPLING
            # ------------------------------------------------

            else:


                probabilities = torch.softmax(

                    logits,

                    dim=-1

                )


                next_token_id = torch.multinomial(

                    probabilities,

                    num_samples=1

                ).item()


        # ----------------------------------------------------
        # ADD PREDICTED TOKEN
        # ----------------------------------------------------

        generated_token_ids.append(

            next_token_id

        )


    # ========================================================
    # SECTION : CONVERT TOKEN IDS BACK TO WORDS
    # ========================================================

    generated_words = []


    for token_id in generated_token_ids:


        word = index_to_word.get(

            token_id,

            UNK_TOKEN

        )


        # Ignore padding tokens.

        if word != PAD_TOKEN:

            generated_words.append(

                word

            )


    # Convert words into final generated text.

    generated_text = " ".join(

        generated_words

    )


    return generated_text


# ============================================================
# SECTION 5 : TEST DETERMINISTIC GENERATION
# ============================================================

print()

print("=" * 70)

print("DETERMINISTIC TEXT GENERATION")

print("=" * 70)


prompt = "artificial intelligence"


generated_text = generate_text(

    prompt=prompt,

    max_new_tokens=20,

    temperature=1.0,

    deterministic=True

)


print()

print("PROMPT:")

print(prompt)


print()

print("GENERATED TEXT:")

print(generated_text)


# ============================================================
# SECTION 6 : TEST TEMPERATURE = 0.7
# ============================================================

print()

print("=" * 70)

print("TEXT GENERATION - TEMPERATURE 0.7")

print("=" * 70)


generated_text = generate_text(

    prompt="machine learning",

    max_new_tokens=20,

    temperature=0.7,

    top_k=10

)


print()

print(generated_text)


# ============================================================
# SECTION 7 : TEST TEMPERATURE = 1.0
# ============================================================

print()

print("=" * 70)

print("TEXT GENERATION - TEMPERATURE 1.0")

print("=" * 70)


generated_text = generate_text(

    prompt="deep learning",

    max_new_tokens=20,

    temperature=1.0,

    top_k=10

)


print()

print(generated_text)


# ============================================================
# SECTION 8 : TEST TEMPERATURE = 1.5
# ============================================================

print()

print("=" * 70)

print("TEXT GENERATION - TEMPERATURE 1.5")

print("=" * 70)


generated_text = generate_text(

    prompt="neural networks",

    max_new_tokens=20,

    temperature=1.5,

    top_k=15

)


print()

print(generated_text)


# ============================================================
# SECTION 9 : TEST MULTIPLE PROMPTS
# ============================================================

test_prompts = [

    "artificial intelligence",

    "machine learning",

    "deep learning",

    "neural networks",

    "transformers",

    "python is"

]


print()

print("=" * 70)

print("MULTIPLE TEXT GENERATION TESTS")

print("=" * 70)


for prompt in test_prompts:


    print()

    print("-" * 70)

    print("PROMPT:")

    print(prompt)


    generated_text = generate_text(

        prompt=prompt,

        max_new_tokens=15,

        temperature=0.8,

        top_k=10

    )


    print()

    print("GENERATED TEXT:")

    print(generated_text)


# ============================================================
# SECTION 10 : COMPARE TEMPERATURE VALUES
# ============================================================

prompt = "artificial intelligence"


temperatures = [

    0.5,

    0.8,

    1.0,

    1.5

]


print()

print("=" * 70)

print("TEMPERATURE COMPARISON")

print("=" * 70)


for temperature_value in temperatures:


    print()

    print(

        f"TEMPERATURE = {temperature_value}"

    )


    generated_text = generate_text(

        prompt=prompt,

        max_new_tokens=15,

        temperature=temperature_value,

        top_k=10

    )


    print()

    print(

        generated_text

    )


# ============================================================
# SECTION 11 : SAVE VOCABULARY
# ============================================================

# Convert vocabulary mappings into files so they can
# be reused later with the trained model.

with open(

    "word_to_index.json",

    "w"

) as file:


    json.dump(

        word_to_index,

        file,

        indent=4

    )


# JSON keys must be strings.

index_to_word_json = {

    str(key): value

    for key, value in index_to_word.items()

}


with open(

    "index_to_word.json",

    "w"

) as file:


    json.dump(

        index_to_word_json,

        file,

        indent=4

    )


print()

print("VOCABULARY FILES SAVED SUCCESSFULLY")


# ============================================================
# SECTION 12 : SAVE MODEL CONFIGURATION
# ============================================================

model_config = {

    "vocab_size": vocab_size,

    "embedding_dim": EMBEDDING_DIM,

    "num_heads": NUM_HEADS,

    "num_layers": NUM_LAYERS,

    "context_length": CONTEXT_LENGTH,

    "dropout": DROPOUT

}


with open(

    "mini_gpt_config.json",

    "w"

) as file:


    json.dump(

        model_config,

        file,

        indent=4

    )


print()

print("MODEL CONFIGURATION SAVED SUCCESSFULLY")


# ============================================================
# SECTION 13 : CREATE SIMPLE INTERACTIVE GENERATION FUNCTION
# ============================================================

def generate_from_prompt(prompt):


    generated_text = generate_text(

        prompt=prompt,

        max_new_tokens=20,

        temperature=0.8,

        top_k=10

    )


    print()

    print("=" * 70)

    print("GPT TEXT GENERATION RESULT")

    print("=" * 70)

    print()

    print("PROMPT:")

    print(prompt)

    print()

    print("GENERATED TEXT:")

    print(generated_text)


# ============================================================
# SECTION 14 : CUSTOM PROMPT TEST
# ============================================================

generate_from_prompt(

    "artificial intelligence is"

)


# ============================================================
# SECTION 15 : DISPLAY PROJECT FILES
# ============================================================

import os


print()

print("=" * 70)

print("SAVED PROJECT FILES")

print("=" * 70)


project_files = [

    MODEL_PATH,

    "word_to_index.json",

    "index_to_word.json",

    "mini_gpt_config.json"

]


for file_name in project_files:


    if os.path.exists(

        file_name

    ):


        print(

            "✔",

            file_name

        )


    else:


        print(

            "✘",

            file_name,

            "NOT FOUND"

        )


# ============================================================
# SECTION 16 : FINAL PROJECT SUMMARY
# ============================================================

print()

print("=" * 70)

print("END-TO-END GPT TEXT GENERATION PROJECT COMPLETED")

print("=" * 70)


print()

print("DATA PIPELINE")

print()

print("✔ Predefined Text Dataset")

print("✔ Text Cleaning")

print("✔ Vocabulary Creation")

print("✔ Word-Level Tokenization")

print("✔ Token ID Conversion")

print("✔ Input Sequence Creation")

print("✔ Next Token Targets")

print()

print("GPT ARCHITECTURE")

print()

print("✔ Token Embeddings")

print("✔ Positional Embeddings")

print("✔ Causal Self-Attention")

print("✔ Multi-Head Attention")

print("✔ Feed Forward Network")

print("✔ Layer Normalization")

print("✔ Residual Connections")

print("✔ Transformer Blocks")

print("✔ Language Modeling Head")

print()

print("TEXT GENERATION")

print()

print("✔ Next Token Prediction")

print("✔ Greedy Generation")

print("✔ Temperature Control")

print("✔ Top-K Sampling")

print("✔ Multiple Prompt Testing")

print()

print("SAVED FILES")

print()

print("✔ mini_gpt_model.pth")

print("✔ word_to_index.json")

print("✔ index_to_word.json")

print("✔ mini_gpt_config.json")

print()

print("PROJECT COMPLETED SUCCESSFULLY!")

print("=" * 70)
