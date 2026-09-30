import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    1D CNN encoder for OTFS I/Q signals.

    Input:
        (batch_size, 2, signal_length)

    Output:
        (batch_size, 128)
    """

    def __init__(self, embedding_dim=128):

        super().__init__()

        self.features = nn.Sequential(

            # First convolution block
            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),

            # Second convolution block
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),

            # Third convolution block
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2)
        )

        # Adaptive pooling allows different signal lengths
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Convert 128 features into embedding
        self.embedding = nn.Linear(
            128,
            embedding_dim
        )

    def forward(self, x):

        # CNN feature extraction
        x = self.features(x)

        # Global average pooling
        x = self.global_pool(x)

        # Remove final dimension
        x = x.squeeze(-1)

        # Create embedding
        x = self.embedding(x)

        return x


if __name__ == "__main__":

    print("===================================")
    print("        CNN ENCODER TEST")
    print("===================================")

    # Create the encoder
    encoder = CNNEncoder(
        embedding_dim=128
    )

    # Create dummy I/Q input
    #
    # Batch size = 4
    # I/Q channels = 2
    # Signal length = 261
    dummy_input = torch.randn(
        4,
        2,
        261
    )

    print("\nInput shape:")
    print(dummy_input.shape)

    # Pass input through encoder
    embeddings = encoder(
        dummy_input
    )

    print("\nEmbedding shape:")
    print(embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0])

    print("\nNumber of embedding features:")
    print(embeddings.shape[1])

    print("\nCNN encoder created successfully!")