"""
WHAT IS SELF-ATTENTION?

Self-Attention allows each token to look at other tokens in the same sentence and decide which ones are important.

It helps the Transformer understand the relationship between words.

For example, in:

"The animal didn't cross the road because it was tired."

Self-Attention helps the model understand what "it" refers to.

The important idea is:

Every token → looks at other tokens → gets contextual information
------------------------------------------------------------------------------------------------------
WHY DO WE NEED IT?

Token Embedding tells us about an individual token.
Positional Encoding tells us where the token is.

But we still need to know:
How is this token related to the other tokens?
For example:
"I love cats"

The meaning of "love" depends on the surrounding words.
Self-Attention allows tokens to exchange information with each other.
-------------------------------------------------------------------------------------------------------

HOW DOES IT WORK?

Let's use a very small example:

I love cats

Suppose we are calculating the new representation of:

love

Self-Attention allows love to look at:

I
love
cats

It assigns different importance to each token:

love → I       0.2
love → love    0.3
love → cats    0.5

So the model says:

"cats" is more important for understanding "love"

Then information from these tokens is combined to create a new representation for love.

The same process happens for every token.

********************************************************************************
Important formula

The complete scaled attention formula will be covered properly in MODULE 07.

For now, just understand the basic idea:

Attention = weighted combination of other token information
"""

# ============================================================
#PRACTICAL IMPLEMENTATION
# ============================================================

import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# 1. Create a small sentence representation
# ------------------------------------------------------------

# Sentence:
# "I love cats"
#
# 3 tokens
# embedding dimension = 4

x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # I
    [0.0, 1.0, 0.0, 1.0],   # love
    [1.0, 1.0, 0.0, 0.0]    # cats
])

print("Input:")
print(x)

print("\nInput shape:")
print(x.shape)

# ------------------------------------------------------------
# 2. Calculate attention scores
# ------------------------------------------------------------

attention_scores = torch.matmul(x, x.T)

print("\nAttention Scores:")
print(attention_scores)

print("\nAttention Scores Shape:")
print(attention_scores.shape)



# ------------------------------------------------------------
# 3. Apply Softmax
# Convert scores into probabilities
# ------------------------------------------------------------

attention_weights = F.softmax(
    attention_scores,
    dim=-1
)

print("\nAttention Weights:")
print(attention_weights)

print("\nAttention Weights Shape:")
print(attention_weights.shape)


# ------------------------------------------------------------
# 4. Weighted combination of token representations
# Create the contextual representation
# ------------------------------------------------------------

output = torch.matmul(
    attention_weights,
    x
)

print("\nSelf-Attention Output:")
print(output)

print("\nOutput Shape:")
print(output.shape)