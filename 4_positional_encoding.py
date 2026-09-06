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

# ------------------------------------------------------------
# 8. Create example token embeddings
# ------------------------------------------------------------

# Suppose:
# batch_size = 1
# sequence_length = 5
# embedding_dim = 6

token_embeddings = torch.randn(
    1,
    sequence_length,
    embedding_dim
)

print("\nToken Embedding Shape:")
print(token_embeddings.shape)


# ------------------------------------------------------------
# 9. Add positional encoding
# ------------------------------------------------------------

# Add a batch dimension to positional encoding.
# [5, 6] → [1, 5, 6]

positional_encoding = positional_encoding.unsqueeze(0)

print("\nPositional Encoding after unsqueeze:")
print(positional_encoding.shape)


# Add token information + position information

transformer_input = token_embeddings + positional_encoding

print("\nTransformer Input:")
print(transformer_input)

print("\nTransformer Input Shape:")
print(transformer_input.shape)