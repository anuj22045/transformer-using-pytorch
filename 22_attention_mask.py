"""
ATTENTION MASK
An Attention Mask tells the Transformer which tokens it is allowed to attend to and which tokens it must ignore.
For example, suppose we have:

I    love    cats

Without a mask, every token can attend to every other token:

I    → I, love, cats
love → I, love, cats
cats → I, love, cats

But sometimes we don't want that.

We can use a mask to say:

"These positions are allowed."
"These positions are blocked."

So the mask basically controls where attention can look.
--------------------------------------------------------------------------------------------------------------------
HOW DOES IT WORK?

Attention normally calculates:
QKᵀ
  ↓
Attention Scores
  ↓
Softmax
  ↓
Attention Weights
The mask is applied to the attention scores before Softmax.
QKᵀ
 ↓
Apply Mask
 ↓
Softmax
 ↓
Attention Weights
 ↓
Weighted Values

For blocked positions, we effectively give them a very large negative value: -∞
Then Softmax makes their probability approximately: 0

----------------------------------------------------------------------------------------------------------------
IMPORTANT CONCEPTS
Attention Mask

General mechanism for controlling attention.

Causal Mask

Blocks future tokens.

Past ✓
Future ✗
Padding Mask

Blocks <PAD> tokens.

Real token ✓
<PAD> ✗
Mask is applied before Softmax
Scores
  ↓
Mask
  ↓
Softmax
"""

import torch


seq_len = 4


# ---------------------------------------------------------
# Create Causal Attention Mask
# ---------------------------------------------------------

attention_mask = torch.triu(
    torch.ones(seq_len, seq_len),
    diagonal=1
).bool()


print("Attention Mask:")
print(attention_mask)

# False → Allowed
# True  → Blocked

# USING MULTI HEAD ATTENTION
import torch
import torch.nn as nn


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

batch_size = 1
seq_len = 4
d_model = 8
num_heads = 2


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

x = torch.randn(
    batch_size,
    seq_len,
    d_model
)


# ---------------------------------------------------------
# Attention Mask
# ---------------------------------------------------------

attention_mask = torch.triu(
    torch.ones(seq_len, seq_len),
    diagonal=1
).bool()


# ---------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------

attention = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    batch_first=True
)


# ---------------------------------------------------------
# Apply Attention Mask
# ---------------------------------------------------------

output, weights = attention(
    x,
    x,
    x,
    attn_mask=attention_mask
)


# ---------------------------------------------------------
# Shapes
# ---------------------------------------------------------

print("Input Shape:")
print(x.shape)

print("\nAttention Mask Shape:")
print(attention_mask.shape)

print("\nOutput Shape:")
print(output.shape)

print("\nAttention Weights Shape:")
print(weights.shape)