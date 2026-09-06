# 7_scaled_dot_product_attention.py
"""

WHAT IS SCALED DOT-PRODUCT ATTENTION?

Scaled Dot-Product Attention is the actual mathematical operation used to calculate attention in a Transformer.

We already learned Q, K, and V.

Now we use them to calculate:

How much attention should each token give to every other token?
----------------------------------------------------------------------------------------------------------------

WHY DO WE NEED IT?

The dot product between Q and K can become very large when the vector dimension is large.

Large values can make Softmax produce extremely sharp probabilities.

So we scale the scores by √d before applying Softmax.

Without scaling:
QKᵀ → potentially very large values

With scaling:
QKᵀ / √d → more stable values

This helps training remain stable.
--------------------------------------------------------------------------------------------------------
HOW DOES IT WORK?

The complete formula is:

                QKᵀ
Attention(Q,K,V) = Softmax(────) V
                √d

Where:

Q = Query
K = Key
V = Value
d = dimension of the Key vectors

The process is:

1. Calculate QKᵀ
2. Divide by √d
3. Apply Softmax
4. Multiply by V
------------------------------------------------------------------------------------------------

IMPORTANT CONCEPTS:
1 — Dot Product

QKᵀ
This gives us the attention scores.
These scores tell us how strongly each Query matches each Key.

2 — Scaling
QKᵀ
────
√d

If: d = 4

then: √d = √4 = 2
So:
[4, 2, 6]
    ↓ divide by 2
[2, 1, 3]


3 — Softmax
Softmax converts the scores into probabilities.
For example:
[2, 1, 3]
might become approximately:
[0.245, 0.090, 0.665]
These values add up to approximately: 1.0

4 — Multiply with V
The attention weights are used to calculate a weighted combination of the Values.
Attention Weights × V
        ↓
        Output
"""
# We will create the keep the element manually and implement complete calculation manually 

# Step 1 — Create Q, K, V
import torch
import math


# ------------------------------------------------------------
# 1. Create Q, K, V
# ------------------------------------------------------------

# 3 tokens
# Each token has 4 dimensions

Q = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # Query for token 1
    [0.0, 1.0, 0.0, 1.0],   # Query for token 2
    [1.0, 1.0, 0.0, 0.0]    # Query for token 3
])

K = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # Key for token 1
    [0.0, 1.0, 0.0, 1.0],   # Key for token 2
    [1.0, 1.0, 0.0, 0.0]    # Key for token 3
])

V = torch.tensor([
    [10.0, 1.0, 2.0, 3.0],  # Value for token 1
    [20.0, 2.0, 3.0, 4.0],  # Value for token 2
    [30.0, 3.0, 4.0, 5.0]   # Value for token 3
])

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Step 2 — Calculate QKᵀ
# ------------------------------------------------------------
# 2. Calculate attention scores
# ------------------------------------------------------------

attention_score=torch.matmul(
    Q, K.T

)
print("\n Attention Scores: ", attention_score)
print("\n Attention score shape: ", attention_score.shape)


# Step 3 — Scale by √d
# ------------------------------------------------------------
# 3. Scale attention scores
# ------------------------------------------------------------
d_k = K.size(-1)
scaled_scores = attention_score / math.sqrt(d_k)

print("\n d_k: ", d_k)

print("\n Square root of d_k: ", math.sqrt(d_k))
print("\n Scaled Attention scores: ", scaled_scores)


# Step 4- Apply Softmax
# ------------------------------------------------------------
# 4. Convert scores into attention weights
# ------------------------------------------------------------
attention_weights = torch.softmax(
    scaled_scores, dim=1
)

print("\n Attention Weights: ", attention_weights)

print("\n Row sums: ", attention_weights.sum(dim=-1))
# Each row should sum approximately to:  1.0

# Step 5 — Multiply with V

# ------------------------------------------------------------
# 5. Calculate final attention output
# ------------------------------------------------------------

output = torch.matmul(
    attention_weights,
    V
)

print("\nFinal Attention Output:")
print(output)

print("\nOutput Shape:")
print(output.shape)
# ------------------------------------------------------------------------------------------------------------
# The complete implementation Summary :
"""
# QK^T
attention_scores = torch.matmul(Q, K.T)

# Scale by sqrt(d_k)
scaled_scores = attention_scores / math.sqrt(d_k)

# Softmax
attention_weights = torch.softmax(
    scaled_scores,
    dim=-1
)

# Weighted sum of V
output = torch.matmul(
    attention_weights,
    V
)
"""