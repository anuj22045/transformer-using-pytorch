# 11_layer_normalization.py
"""
WHAT IS LAYER NORMALIZATION?

Layer Normalization (LayerNorm) is a technique used in Transformers to normalize the values of a token's representation.
In simple words:
Layer Normalization keeps the values inside the network stable and makes training easier.
Suppose one token has this representation:
[10, 2, 30, 5]
The values are very different in scale.
LayerNorm normalizes them so the representation becomes more stable.
-----------------------------------------------------------------------------------------------------------------------------
WHY DO WE NEED IT?

Transformers contain many operations:
Attention
   ↓
Feed Forward
   ↓
Attention
   ↓
Feed Forward
   ↓
...

As information passes through these layers, values can become too large or too small.
This can make training:
unstable
slower
harder to optimize
Layer Normalization helps keep the activations under control.
-----------------------------------------------------------------------------------------------------------------------------

HOW DOES IT WORK?

Suppose one token has:

x = [2, 4, 6, 8]

LayerNorm looks at the values inside this token representation.

Step 1 → Calculate Mean
Mean = (2 + 4 + 6 + 8) / 4
    = 5
    
Step 2 → Calculate Variance
It measures how far the values are from the mean.

Step 3 → Normalize
Conceptually:
normalized = (x - mean) / sqrt(variance + ε)
ε is a very small value added to avoid division by zero.
Step 4 → Learnable Parameters

LayerNorm also has two learnable parameters:
γ → scale
β → shift

Final:
output = γ × normalized + β

So the complete idea is:
Input
  ↓
Calculate Mean
  ↓
Calculate Variance
  ↓
Normalize
  ↓
Scale (γ)
  ↓
Shift (β)
  ↓
Output
"""

# -----------------------------------------------------------------------------
# PRACTICAL IMPLEMENTATION
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn

# ---------------------------------------------------------
# Input
# 3 tokens, d_model = 4
# ---------------------------------------------------------

x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0]
])

# ---------------------------------------------------------
# Simulated attention output
# ---------------------------------------------------------

attention_output = torch.tensor([
    [0.5, 0.2, 0.3, 0.4],
    [0.1, 0.4, 0.2, 0.3],
    [0.3, 0.5, 0.1, 0.2]
])

# ---------------------------------------------------------
# Residual Connection
# ---------------------------------------------------------

residual_output = x + attention_output

# ---------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------

layer_norm = nn.LayerNorm(4)

normalized_output = layer_norm(residual_output)

print("Input shape:")
print(x.shape)

print("\nAfter Residual Connection:")
print(residual_output)

print("\nAfter LayerNorm:")
print(normalized_output)

print("\nFinal shape:")
print(normalized_output.shape)


"""
-----------------------------------------------------------------------------
MEMORY NOTE
-----------------------------------------------------------------------------
I need stable token representations → Layer Normalization
I need to normalize each token's features → LayerNorm
I need learnable scaling and shifting → γ and β
I need to stabilize Transformer training → LayerNorm
"""