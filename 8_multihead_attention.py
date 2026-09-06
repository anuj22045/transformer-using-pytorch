# 8_multihead_attention.py
"""
WHAT IS MULTI-HEAD ATTENTION?

Multi-Head Attention means running multiple attention mechanisms (heads) in parallel.
Instead of having one attention operation understand relationships between tokens, we create several heads.
Each head can learn to focus on different types of relationships.
Then all the heads are combined to produce the final output.
----------------------------------------------------------------------------------------------------------------------

WHY DO WE NEED IT?

One attention mechanism may focus on one type of relationship.
For example, one head might learn:

subject ↔ verb
while another might learn:
verb ↔ object
and another might focus on:
nearby words

Multiple heads allow the Transformer to capture different relationships at the same time.
-----------------------------------------------------------------------------------------------------------------------
IMPORTANT CONCEPTS:
1. Multiple Heads
If:
d_model = 8
num_heads = 2
we split the representation into:
head dimension = 8 / 2 = 4

So:
8-dimensional representation
        ↓
┌──────┴──────┐
↓             ↓
Head 1        Head 2
4 dimensions  4 dimensions


2. Each Head Performs Attention
Each head performs:
QKᵀ
────
√d

followed by: Softmax → × V

So Multi-Head Attention is basically:
Several Scaled Dot-Product Attentions
                ↓
            Concatenate
                ↓
        Linear Projection

3. Concatenation
Suppose:

Head 1 → [3, 4]
Head 2 → [3, 4]

After concatenation: [3, 8]
The dimensions are joined together.

4. Final Linear Layer
After concatenating all heads, we pass the result through a learnable linear projection:
Concatenated Heads
       ↓
     Linear
       ↓
     Output
"""

# We will NOT use:
# nn.MultiheadAttention
# We will build the important calculations ourselves.

import torch 
import torch.nn as nn
import math

# ------------------------------------------------------------
# 1. Define input
# ------------------------------------------------------------

# 3 tokens
# d_model = 8

X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
])

print("Input shape:")
print(X.shape)

# Step 2 — Define number of heads
# ------------------------------------------------------------
# 2. Define Multi-Head Attention parameters
# ------------------------------------------------------------

d_model = 8
num_heads = 2

#Every head get parts of d_model

head_dim = d_model // num_heads

print("\n d_model: ", d_model)
print("num_heads: ", num_heads)
print("head_dim: ", head_dim)

# Step 3 — Create Q, K, V projections
# ------------------------------------------------------------
# 3. Create projection matrices
# ------------------------------------------------------------

WQ = nn.Linear(d_model, d_model, bias=False)
WK = nn.Linear(d_model, d_model, bias=False)
WV = nn.Linear(d_model, d_model, bias=False)

Q = WQ(X)
K = WK(X)
V = WV(X)

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Step 4 — Split into heads
# ------------------------------------------------------------
# 4. Split Q, K, V into multiple heads
# ------------------------------------------------------------
# We need to transform: [3, 8]

# into: [2, 3, 4]

# Meaning:
# 2 → number of heads
# 3 → sequence length
# 4 → head dimension

Q = Q.view(
    -1,
    num_heads,
    head_dim
).transpose(0, 1)

K = K.view(
    -1,
    num_heads,
    head_dim
).transpose(0, 1)

V = V.view(
    -1,
    num_heads,
    head_dim
).transpose(0, 1)

print("\nAfter splitting into heads:")

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Step 5 — Calculate attention scores
# ------------------------------------------------------------
# 5. Calculate attention scores for every head
# ------------------------------------------------------------

attention_scores = torch.matmul(
    Q,
    K.transpose(-2, -1)
)
print("\nAttention Scores Shape:")
print(attention_scores.shape)


# Step 6 — Scale the scores
# ------------------------------------------------------------
# 6. Scale attention scores
# ------------------------------------------------------------

scaled_scores = attention_scores / math.sqrt(head_dim)

print("\nScaled Scores Shape:")
print(scaled_scores.shape)


# Step 7 — Softmax
# ------------------------------------------------------------
# 7. Convert scores into attention weights
# ------------------------------------------------------------

attention_weights = torch.softmax(
    scaled_scores,
    dim=-1
)

print("\nAttention Weights Shape:")
print(attention_weights.shape)

print("\nAttention Weights:")
print(attention_weights)


# Step 8 — Multiply by V
# ------------------------------------------------------------
# 8. Calculate attention output for each head
# ------------------------------------------------------------

head_outputs = torch.matmul(
    attention_weights,
    V
)

print("\nHead Outputs Shape:")
print(head_outputs.shape)


# Step 9 — Concatenate the heads
# ------------------------------------------------------------
# 9. Concatenate all heads
# ------------------------------------------------------------

head_outputs = head_outputs.transpose(
    0,
    1
)

print("\nAfter transpose:")
print(head_outputs.shape)


concatenated = head_outputs.contiguous().view(
    -1,
    d_model
)

print("\nConcatenated Heads Shape:")
print(concatenated.shape)

# Step 10 — Final Linear Projection
# ------------------------------------------------------------
# 10. Final output projection
# ------------------------------------------------------------

output_projection = nn.Linear(
    d_model,
    d_model
)

output = output_projection(
    concatenated
)

print("\nFinal Multi-Head Attention Output:")
print(output)

print("\nFinal Output Shape:")
print(output.shape)