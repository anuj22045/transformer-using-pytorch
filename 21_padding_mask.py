"""
DEFINITION

A Padding Mask tells the Transformer which tokens are just padding and should be ignored during attention.

Let's say we have sentences with different lengths:

Sentence 1:
I love cats

Sentence 2:
I love cats very much

A batch needs the same sequence length, so we add <PAD> tokens:

I    love    cats    <PAD>   <PAD>

I    love    cats    very    much
The problem is that <PAD> doesn't contain useful information.
We don't want the model to think:
"Maybe <PAD> is an important word."
So we create a Padding Mask that tells attention:
Real token → Look at it ✓
<PAD>      → Ignore it ✗
"""

import torch


# ---------------------------------------------------------
# Token IDs
# ---------------------------------------------------------

# 0 = <PAD>

tokens = torch.tensor([
    [1, 2, 3, 0, 0],
    [1, 2, 3, 4, 5]
])


# ---------------------------------------------------------
# Create Padding Mask
# ---------------------------------------------------------

padding_mask = tokens == 0


# ---------------------------------------------------------
# Print Results
# ---------------------------------------------------------

print("Tokens:")
print(tokens)

print("\nPadding Mask:")
print(padding_mask)

print("\nPadding Mask Shape:")
print(padding_mask.shape)