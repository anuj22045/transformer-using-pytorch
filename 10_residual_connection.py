# 10_residual_connection.py
"""
WHAT IS RESIDUAL CONNECTION?

A Residual Connection allows the original input to skip over a layer and be added directly to its output.
Instead of:
Input
  ↓
Layer
  ↓
Output

we do:
Input ───────────────┐
  ↓                  ↓
Layer             Addition
  ↓                  ↑
  └──────────────────┘
        ↓
    Output

The basic formula is:
Output = x + Layer(x)
---------------------------------------------------------------------------------------------------------

WHY DO WE NEED IT?

Transformers contain many layers.
As the network becomes deeper, training can become harder because useful information can get lost or gradients
can become difficult to propagate.
Residual connections provide a direct path for information and gradients to flow through the network.

HOW DOES IT WORK?
Suppose the input is: x = [1, 2, 3]

A layer processes it:
Layer(x) = [0.5, 1, 2]
Instead of using only: [0.5, 1, 2]

we add the original input:
x + Layer(x)
[1, 2, 3]
+
[0.5, 1, 2]
=
[1.5, 3, 5]

So:
Input
 ↓
Layer
 ↓
Layer Output
 ↓
   +
 ↑
Input
 ↓
Residual Output
----------------------------------------------------------------------------------------------

"""
# -----------------------------------------------------------------------------
# PRACTICAL IMPLEMENTATION
# -----------------------------------------------------------------------------
# We'll first implement the residual connection manually.

import torch
import torch.nn as nn


# ------------------------------------------------------------
# 1. Create input
# ------------------------------------------------------------

# 3 tokens
# Each token has 4 features

x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0]
])

print("Input:")
print(x)

print("\nInput shape:")
print(x.shape)

# ------------------------------------------------------------
# 2. Create a layer
# ------------------------------------------------------------

layer = nn.Linear(
    4,
    4
)

layer_output = layer(x)

print("\nLayer Output:")
print(layer_output)

print("\nLayer Output Shape:")
print(layer_output.shape)

# ------------------------------------------------------------
# 3. Residual Connection
# ------------------------------------------------------------

residual_output = x + layer_output

print("\nResidual Output:")
print(residual_output)

print("\nResidual Output Shape:")
print(residual_output.shape)

# ------------------------------------------------------------
# 4. Verify shapes
# ------------------------------------------------------------

print("\nShape comparison:")
print("Input:", x.shape)
print("Layer Output:", layer_output.shape)
print("Residual Output:", residual_output.shape)

# ------------------------------------------------------------
# 5. Transformer-style residual operation
# ------------------------------------------------------------

attention_output = layer(x)

# Add the original input back
encoder_representation = x + attention_output

print("\nTransformer-style Residual Output:")
print(encoder_representation)

print("\nShape:")
print(encoder_representation.shape)

"""
-----------------------------------------------------------------------------
MEMORY NOTE
-----------------------------------------------------------------------------

I need the original information to bypass a layer
→ Skip Connection

I need to combine original + processed information
→ x + F(x)

I need better information and gradient flow
→ Residual Connection
"""