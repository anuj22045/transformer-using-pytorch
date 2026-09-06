# 9_feed_forward_network.py
"""
WHAT IS FEED FORWARD NETWORK?

The Feed Forward Network (FFN) is a small neural network applied to each token independently after Multi-Head Attention.
Attention allows tokens to communicate with each other.
The FFN then processes the information of each token individually and learns a more useful representation.
-----------------------------------------------------------------------------------------------------------------------------

WHY DO WE NEED IT?

Multi-Head Attention is mainly responsible for relationships between tokens.
But after attention, we need a neural network to further process and transform the information.
The FFN gives the model additional non-linear learning capacity.
think:
Attention → Understand relationships
FFN       → Process the information
-------------------------------------------------------------------------------------------------------------------------------

HOW DOES IT WORK?

Suppose after attention we have: [0.2, 0.5, 0.1, 0.8]
The FFN first expands the vector:
4 dimensions
    ↓
Linear
    ↓
8 dimensions

Then activation:
[...]
    ↓
ReLU / GELU

Then it projects it back:
8 dimensions
    ↓
Linear
    ↓
4 dimensions

So:
[4]
 ↓
[8]
 ↓
[4]

The important point is that the FFN is applied separately to each token.
"""

# -----------------------------------------------------------------------------
# PRACTICAL IMPLEMENTATION
# -----------------------------------------------------------------------------

# ============================================================
# MODULE 09: FEED FORWARD NETWORK
# ============================================================

import torch
import torch.nn as nn


# ------------------------------------------------------------
# 1. Create input
# ------------------------------------------------------------

# 3 tokens
# Each token has 4 features

x = torch.tensor([
    [0.2, 0.5, 0.1, 0.8],   # Token 1
    [0.4, 0.3, 0.7, 0.2],   # Token 2
    [0.9, 0.1, 0.5, 0.6]    # Token 3
])

print("Input:")
print(x)

print("\nInput shape:")
print(x.shape)

d_model =4
d_ff = 8

linear1 = nn.Linear(
    d_model, d_ff
)

hidden = linear1(x)

print("\n After first linear layer: ", hidden)

print("\n Shape: ", hidden.shape)


# Step 2 — First Linear Layer
# ------------------------------------------------------------
# 2. First Linear Layer
# ------------------------------------------------------------

d_model = 4
d_ff = 8

linear1 = nn.Linear(
    d_model,
    d_ff
)

hidden = linear1(x)

print("\nAfter first Linear Layer:")
print(hidden)

print("\nShape:")
print(hidden.shape)

# Step 3 — Apply ReLU
# ------------------------------------------------------------
# 3. Apply ReLU activation
# ------------------------------------------------------------

relu_output = torch.relu(hidden)

print("\n After reLU: ", relu_output)

print("\n Shape: ", relu_output.shape)

# Step 4 — Second Linear Layer
# ------------------------------------------------------------
# 4. Project back to d_model
# ------------------------------------------------------------
linear2 = nn.Linear(
    d_ff, d_model
)
output = linear2(relu_output)

print("\n Final FFN Output: ", output)

print("\n Final Output Shape: ", output.shape)


"""
MEMORY NOTE

I need tokens to understand relationships
→ Attention

I need to process each token's information
→ Feed Forward Network

I need more learning capacity
→ Linear → Activation → Linear
"""