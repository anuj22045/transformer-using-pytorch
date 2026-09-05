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
print("English tokens: ", english_tokens)

print("SHape: ", english_tokens.shape)

# ------------------------------------------------------------
# 2. Token Embedding
# ------------------------------------------------------------

vocab_size = 10
embedding_dim = 4

embedding = nn.Embedding(vocab_size, embedding_dim)
encoder_input = embedding(english_tokens)

print("\n Encoder input shape: ", encoder_input.shape)

# ------------------------------------------------------------
# 3. Simple Encoder
# ------------------------------------------------------------

encoder_layer = nn.TransformerEncoderLayer(
    d_model = embedding_dim, nhead=2, batch_first=True
)

encoder = nn.TransformerEncoder(
    encoder_layer, num_layers=1
)

encoder_output = encoder(encoder_input)

print("\n Encoder output: ", encoder_output)

print("Encoder output shape:")
print(encoder_output.shape)

# ------------------------------------------------------------
# 4. French tokens for Decoder
# ------------------------------------------------------------
# Suppose:
# <START> -> 0
# Je      -> 1
# t'aime  -> 2

french_tokens = torch.tensor([[0, 1, 2]])
decoder_input = embedding(french_tokens)

print("\n Decoder input shape: ", decoder_input.shape)

# ------------------------------------------------------------
# 5. Simple Decoder
# ------------------------------------------------------------

decoder_layer = nn.TransformerDecoderLayer(
    d_model = embedding_dim, nhead=2, batch_first=True
)

decoder = nn.TransformerDecoder(
    decoder_layer, num_layers=1
)

decoder_output = decoder(
    decoder_input, encoder_output
)

print("\nDecoder output:")
print(decoder_output)

print("Decoder output shape:")
print(decoder_output.shape)