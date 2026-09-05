# 2_encoder_decoder.py
"""
WHAT IS ENCODER VS DECODER?

A Transformer has two major parts:

Encoder → understands the input sentence.

Decoder → generates the output sentence.
---------------------------------------------------------------------------------
HOW DOES IT WORK?
Suppose:

English: I love cats

->Encoder
It receives: I → love → cats

and produces contextual representations:
[vector] [vector] [vector]

These vectors contain information about the sentence.

->Decoder

The decoder uses that information to generate:

<START> → J' → aime → les → chats → <END>

So the decoder generates the translation step-by-step.
-----------------------------------------------------------------------------------
WHERE IS IT USED?

The classic Encoder-Decoder Transformer is especially useful for sequence-to-sequence tasks such as:

Machine translation
Text summarization
Sequence transformation

"""
# ============================================================
# MODULE 02: ENCODER VS DECODER
# ============================================================
import torch
import torch.nn as nn

# ------------------------------------------------------------
# 1. Example input
# ------------------------------------------------------------

# English tokens:
# I    -> 1
# love -> 2
# you  -> 3

english_tokens = torch.tensor([[1,2,3]])


