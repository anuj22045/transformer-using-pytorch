"""
WHAT IS TRANSFORMER?
A Transformer is a deep learning architecture designed mainly to work with sequences such as text.
Instead of processing words one-by-one like an RNN, it can process the whole sequence together.

Its key idea is Attention — allowing words to understand their relationship with other words.

---------------------------------------------------------------------------------------------------------
WHY DO WE NEED IT?
RNN/LSTM processes sequences step-by-step, which can make learning long-range relationships difficult.
Transformers use attention, so a word can directly look at other relevant words in the sentence.
---------------------------------------------------------------------------------------------------------

HOW DOES IT WORK?
At a high level:

English Sentence
    ↓
Tokens
    ↓
Embedding
    ↓
Positional Encoding
    ↓
Encoder
    ↓
Decoder
    ↓
French Tokens
    ↓
French Sentence
----------------------------------------------------------------------------------------------------------
IMPORTANT CONCEPTS:

Remember only these for now:

Encoder → understands the input sentence
Decoder → generates the output sentence
Attention → finds relationships between tokens
Embedding → converts token IDs into vectors
Positional Encoding → tells the model token positions
"""

# ============================================================
# MODULE 01: TRANSFORMER ARCHITECTURE OVERVIEW
# ============================================================

import torch 
import torch.nn as nn

# ------------------------------------------------------------
# 1. Example token IDs
# ------------------------------------------------------------

# Suppose our English sentence is:
# "I love cats"

# After tokenization:
# I    -> 1
# love -> 2
# cats -> 3

input_tokens = torch.tensor([[1,2,3]])

print("INput tokens: ", input_tokens)

print("Input Shape: ", input_tokens.shape)

# ------------------------------------------------------------
# 2. Token Embedding
# ------------------------------------------------------------

vocab_size = 10
embedding_dim = 4

embedding = nn.Embedding(vocab_size, embedding_dim)

embedded = embedding(input_tokens)
print("\n Embedded tokens: ", embedded)

print("Embedding shape: ", embedded.shape)

#[batch_size, sequence_length, embedding_dim]

# ------------------------------------------------------------
# 3. Positional Encoding
# ------------------------------------------------------------
# For now, we use a simple positional vector.
# We will implement real positional encoding
# in MODULE 04.

position = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 2.0]
])

position = position.unsqueeze(0)

print("\n position shape: ", position.shape)

#Add token information + position information

encoder_input = embedded + position

print("Encoder input shape: ", encoder_input.shape)


# ------------------------------------------------------------
# 4. Simple Encoder placeholder
# ------------------------------------------------------------

encoder = nn.TransformerEncoderLayer(
    d_model=embedding_dim,
    nhead=2,
    batch_first=True
)

encoder_output = encoder(encoder_input)

print("\nEncoder output shape:", encoder_output.shape)


# ------------------------------------------------------------
# Overall flow
# ------------------------------------------------------------

print("\nTransformer flow:")
print("Tokens")
print("  ↓")
print("Embedding")
print("  ↓")
print("Positional Encoding")
print("  ↓")
print("Encoder")
print("  ↓")
print("Decoder")
print("  ↓")
print("French Output")