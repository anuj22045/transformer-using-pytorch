# 18_linear_vocabulary_logits.py
"""
WHAT IS LINEAR + VOCABULARY LOGITS?

The Decoder produces a vector representation for every target token.

For example:

Decoder Output
[batch, sequence_length, d_model]

But this vector does not directly tell us which word to predict.

We need to convert it into scores for every token in our vocabulary.
---------------------------------------------------------------------------------------------------
That's the job of the Linear Layer.

Decoder Output
    ↓
Linear Layer
    ↓
One score for every vocabulary token
    ↓
Vocabulary Logits

For example, suppose our French vocabulary contains
["<PAD>", "<START>", "Je", "aime", "les", "chats", "<END>"]

Vocabulary size:
vocab_size = 7

The Linear Layer produces:
[0.2, -1.1, 3.5, 1.2, 0.4, 2.1, -0.5]
These numbers are called logits.
--------------------------------------------------------------------------------------------------------------

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

batch_size = 2
sequence_length = 3

d_model = 8
vocab_size = 7


# ---------------------------------------------------------
# Simulated Decoder Output
# ---------------------------------------------------------

decoder_output = torch.randn(
    batch_size,
    sequence_length,
    d_model
)

print("Decoder Output Shape:")
print(decoder_output.shape)


# ---------------------------------------------------------
# Linear Layer
# d_model → vocab_size
# ---------------------------------------------------------

output_layer = nn.Linear(
    d_model,
    vocab_size
)


# ---------------------------------------------------------
# Calculate Vocabulary Logits
# ---------------------------------------------------------

logits = output_layer(
    decoder_output
)

print("\nVocabulary Logits Shape:")
print(logits.shape)

print("\nVocabulary Logits:")
print(logits)


# ---------------------------------------------------------
# Convert Logits into Probabilities
# ---------------------------------------------------------

probabilities = F.softmax(
    logits,
    dim=-1
)

print("\nProbabilities Shape:")
print(probabilities.shape)


# ---------------------------------------------------------
# Find Predicted Token
# ---------------------------------------------------------

predicted_tokens = torch.argmax(
    probabilities,
    dim=-1
)

print("\nPredicted Token IDs:")
print(predicted_tokens)

print("\nPredicted Token IDs Shape:")
print(predicted_tokens.shape)

"""
-----------------------------------------------------------------------------
MEMORY NOTE
-----------------------------------------------------------------------------
I need vocabulary scores from Decoder output → Linear Layer
I need raw prediction scores → Logits
I need probabilities → Softmax
I need the most likely token → Argmax
"""