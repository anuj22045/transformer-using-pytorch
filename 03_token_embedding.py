# 3_token_embedding.py
"""
WHAT IS TOKEN EMBEDDING?

Token Embedding converts token IDs into numerical vectors.

A Transformer cannot directly understand:
I → 1
love → 2
cats → 3

So each token ID is converted into a vector.

Example:
1 → [0.2, 0.7, 0.1, 0.5]
2 → [0.8, 0.3, 0.6, 0.2]
------------------------------------------------------------------------

IMPORTANT CONCEPTS:

Embedding Matrix

If:
vocab_size = 10
embedding_dim = 4

then:
Embedding Matrix shape = [10, 4]

Meaning:
10 tokens x 4 numbers per token

Important formula
Token ID → Row of Embedding Matrix → Vector

"""

# ============================================================
# MODULE 03: TOKEN EMBEDDING
# ============================================================

import torch
import torch.nn as nn

# ------------------------------------------------------------
# 1. Define vocabulary and embedding size
# ------------------------------------------------------------

vocab_size = 10
embedding_dim = 4

# Create embedding layer
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# ------------------------------------------------------------
# 2. Look at the embedding matrix
# ------------------------------------------------------------

print("Embedding Matrix:")
print(embedding.weight)

print("\nEmbedding Matrix Shape:")
print(embedding.weight.shape)


# ------------------------------------------------------------
# 3. Create token IDs
# ------------------------------------------------------------

# Suppose:
# I    -> 1
# love -> 2
# cats -> 3

tokens = torch.tensor([[1, 2, 3]])

print("\nToken IDs:")
print(tokens)

print("Token Shape:")
print(tokens.shape)


# ------------------------------------------------------------
# 4. Convert token IDs into embeddings
# ------------------------------------------------------------

embedded_tokens = embedding(tokens)

print("\nEmbedded Tokens:")
print(embedded_tokens)

print("\nEmbedded Tokens Shape:")
print(embedded_tokens.shape)