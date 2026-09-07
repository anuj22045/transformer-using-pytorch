"""
WHAT IS CROSS-ATTENTION?

Cross-Attention allows the Transformer Decoder to look at the Encoder's output while generating the target sequence.
The key idea is:
Decoder asks: "Which parts of the Encoder's input are important for generating my next token?
So Cross-Attention creates a connection between:
Encoder → Decoder
--------------------------------------------------------------------------------------------------------------------

WHY IT IS NEEDED
Imagine the Decoder has already generated:

<START> Je

Now it needs to generate the next French word.

It needs to understand the English input:

I love cats

The Decoder should be able to ask:

"Which English words are important for my next prediction?"

Cross-Attention allows this.

--------------------------------------------------------------------------------------------------------------
IMPORTANT CONCEPTS ⭐⭐⭐
1. Q comes from Decoder
This is the most important point:
Q ← Decoder

The Decoder asks:
"What am I looking for?"

2. K and V come from Encoder
K ← Encoder
V ← Encoder
The Encoder provides information about the source sentence.

Example :
English:
"I love cats"
        ↓ Encoder

Encoder understands:
"I", "love", "cats"

        ↓
Decoder generates:
"J'aime les chats"

While the Decoder is generating a word, it needs to ask the Encoder:
"Which part of the input sentence should I pay attention to?"
That's what Cross-Attention does.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

batch_size = 2

encoder_sequence_length = 5
decoder_sequence_length = 3

d_model = 8


# ---------------------------------------------------------
# Encoder Output
# ---------------------------------------------------------

encoder_output = torch.randn(
    batch_size,
    encoder_sequence_length,
    d_model
)


# ---------------------------------------------------------
# Decoder Output
# ---------------------------------------------------------

decoder_output = torch.randn(
    batch_size,
    decoder_sequence_length,
    d_model
)


print("Encoder Output Shape:")
print(encoder_output.shape)

print("\nDecoder Output Shape:")
print(decoder_output.shape)


# ---------------------------------------------------------
# Linear Layers for Q, K, V
# ---------------------------------------------------------

WQ = nn.Linear(d_model, d_model)
WK = nn.Linear(d_model, d_model)
WV = nn.Linear(d_model, d_model)


# ---------------------------------------------------------
# Create Q, K, V
# ---------------------------------------------------------

Q = WQ(decoder_output)

K = WK(encoder_output)

V = WV(encoder_output)


print("\nQ Shape:")
print(Q.shape)

print("\nK Shape:")
print(K.shape)

print("\nV Shape:")
print(V.shape)


# ---------------------------------------------------------
# Calculate Attention Scores
# ---------------------------------------------------------

scores = torch.matmul(
    Q,
    K.transpose(-2, -1)
)

print("\nAttention Scores Shape:")
print(scores.shape)


# ---------------------------------------------------------
# Scale Attention Scores
# ---------------------------------------------------------

d_k = K.size(-1)

scores = scores / torch.sqrt(
    torch.tensor(d_k, dtype=torch.float32)
)


# ---------------------------------------------------------
# Softmax
# ---------------------------------------------------------

attention_weights = F.softmax(
    scores,
    dim=-1
)

print("\nAttention Weights Shape:")
print(attention_weights.shape)


# ---------------------------------------------------------
# Weighted Values
# ---------------------------------------------------------

output = torch.matmul(
    attention_weights,
    V
)

print("\nCross-Attention Output Shape:")
print(output.shape)

"""
-----------------------------------------------------------------------------
MEMORY NOTE
-----------------------------------------------------------------------------
I need the Decoder to access Encoder information → Cross-Attention
I need the Decoder's current context → Query
I need source information for matching → Encoder Key
I need source information to retrieve → Encoder Value
"""