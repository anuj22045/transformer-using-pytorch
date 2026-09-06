# 4_positional_encoding.py
"""
WHAT IS POSITIONAL ENCODING?

Positional Encoding adds position information to each token embedding.

A Transformer processes all tokens at the same time.

Because of this, the Transformer itself does not automatically know the order of the words.
-----------------------------------------------------------------------------------------------------
WHY DO WE NEED IT?

Token Embedding tells the Transformer what the token is.

But it doesn't tell the Transformer where the token is.

For example:

I     → position 0
love  → position 1
cats  → position 2

Positional Encoding gives the model this position information.
------------------------------------------------------------------------------------------------------
HOW DOES IT WORK?

Suppose we have:

I love cats

After embedding:

I     → [0.2, 0.5, 0.1, 0.8]
love  → [0.7, 0.3, 0.6, 0.2]
cats  → [0.4, 0.9, 0.2, 0.5]

We create a positional vector for each position:

position 0 → [0.0, 1.0, 0.0, 1.0]
position 1 → [0.84, 0.54, 0.84, 0.54]
position 2 → [0.91, -0.42, 0.91, -0.42]

Then we add them:

Token Embedding
    +
Positional Encoding
    ↓
Transformer Input

For example:

I embedding([0.2, 0.5, 0.1, 0.8]) +  Position 0 ([0.0, 1.0, 0.0, 1.0]) = [0.2, 1.5, 0.1, 1.8]
---------------------------------------------------------------------------------------------------

IMPORTANT CONCEPTS:
1. Position
Every token gets a position:

I     → 0
love  → 1
cats  → 2

2. Add, don't concatenate

The Transformer normally uses:
Embedding + Positional Encoding


3. Sinusoidal Positional Encoding
The original Transformer uses sine and cosine functions.

The formulas are:
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
Sine and cosine provide different patterns for different positions 
and allow the model to represent positional relationships across different scales.

4. Shape
If:
sequence_length = 3
embedding_dim = 4

then:
Positional Encoding
shape = [3, 4]

"""

# ============================================================
# MODULE 04: POSITIONAL ENCODING
# ============================================================

import torch
import math


# ------------------------------------------------------------
# 1. Define basic parameters
# ------------------------------------------------------------

sequence_length = 5
embedding_dim = 6


# ------------------------------------------------------------
# 2. Create a positional encoding matrix
# ------------------------------------------------------------

# Shape:
# [sequence_length, embedding_dim]

positional_encoding = torch.zeros(
    sequence_length,
    embedding_dim
)


# ------------------------------------------------------------
# 3. Create position values
# ------------------------------------------------------------

# Positions:
# 0, 1, 2, 3, 4

position = torch.arange(
    sequence_length,
    dtype=torch.float
).unsqueeze(1)

print("Position shape:")
print(position.shape)

print("\nPosition:")
print(position)


# ------------------------------------------------------------
# 4. Create the division term
# ------------------------------------------------------------

# This is part of the original Transformer
# sinusoidal positional encoding formula.

div_term = torch.exp(
    torch.arange(
        0,
        embedding_dim,
        2
    ).float()
    * (-math.log(10000.0) / embedding_dim)
)

print("\nDivision term:")
print(div_term)

print("\nDivision term shape:")
print(div_term.shape)


# ------------------------------------------------------------
# 5. Apply Sine to even dimensions
# ------------------------------------------------------------

# Even dimensions:
# 0, 2, 4, ...

positional_encoding[:, 0::2] = torch.sin(
    position * div_term
)


# ------------------------------------------------------------
# 6. Apply Cosine to odd dimensions
# ------------------------------------------------------------

# Odd dimensions:
# 1, 3, 5, ...

positional_encoding[:, 1::2] = torch.cos(
    position * div_term
)


# ------------------------------------------------------------
# 7. Print positional encoding
# ------------------------------------------------------------

print("\nPositional Encoding:")
print(positional_encoding)

print("\nPositional Encoding Shape:")
print(positional_encoding.shape)

"""
-----------------------------------------------------------------------------
MEMORY NOTE
-----------------------------------------------------------------------------

I need the Transformer to understand token order
→ Positional Encoding

I need position information for every token
→ Position vectors

I need to combine token meaning and position
→ Embedding + Positional Encoding
"""