"""
COMPLETE TRANSFORMER
A Complete Transformer combines the Encoder, Decoder, Attention, Feed Forward, Residual Connections, LayerNorm, masking, and Linear layer into one complete architecture.
"""

import torch
import torch.nn as nn

from embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):

    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        d_model=128,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        dropout=0.1,
        max_length=100,
        pad_id=0
    ):
        super().__init__()

        self.pad_id = pad_id

        # Token embeddings
        self.source_embedding = TokenEmbedding(
            source_vocab_size,
            d_model
        )

        self.target_embedding = TokenEmbedding(
            target_vocab_size,
            d_model
        )

        # Positional encoding
        self.source_position = PositionalEncoding(
            d_model,
            max_length
        )

        self.target_position = PositionalEncoding(
            d_model,
            max_length
        )

        # Encoder
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )

        # Decoder
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )

        # Final output projection
        self.output_layer = nn.Linear(
            d_model,
            target_vocab_size
        )

    def create_source_mask(self, source):

        mask = (source != self.pad_id)

        mask = mask.unsqueeze(1).unsqueeze(2)

        return mask

    def create_target_mask(self, target):

        padding_mask = (target != self.pad_id)

        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        sequence_length = target.size(1)

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=target.device,
                dtype=torch.bool
            )
        )

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        mask = padding_mask & causal_mask

        return mask

    def forward(self, source, target):

        # Create masks
        source_mask = self.create_source_mask(source)

        target_mask = self.create_target_mask(target)

        # Source embedding + positional encoding
        source = self.source_embedding(source)

        source = self.source_position(source)

        # Target embedding + positional encoding
        target = self.target_embedding(target)

        target = self.target_position(target)

        # Encoder
        encoder_output = self.encoder(
            source,
            source_mask
        )

        # Decoder
        decoder_output = self.decoder(
            target,
            encoder_output,
            source_mask,
            target_mask
        )

        # Convert decoder representations
        # into vocabulary logits
        output = self.output_layer(
            decoder_output
        )

        return output


if __name__ == "__main__":

    source_vocab_size = 1000
    target_vocab_size = 1200

    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 4

    batch_size = 2
    source_length = 7
    target_length = 6

    source = torch.randint(
        1,
        source_vocab_size,
        (
            batch_size,
            source_length
        )
    )

    target = torch.randint(
        1,
        target_vocab_size,
        (
            batch_size,
            target_length
        )
    )

    model = Transformer(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )

    output = model(
        source,
        target
    )

    print("Source shape:", source.shape)
    print("Target shape:", target.shape)
    print("Output shape:", output.shape)