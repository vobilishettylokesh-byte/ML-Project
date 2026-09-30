import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import CNNEncoder


class ProjectionHead(nn.Module):
    """
    Projection head used for contrastive learning.

    Converts the encoder embedding into a smaller
    representation used by the contrastive loss.
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=128,
        output_dim=64
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )

    def forward(self, x):

        return self.network(x)


class ContrastiveModel(nn.Module):
    """
    Complete self-supervised contrastive learning model.

    Signal
       ↓
    CNN Encoder
       ↓
    128-D embedding
       ↓
    Projection Head
       ↓
    64-D projection
    """

    def __init__(
        self,
        embedding_dim=128,
        projection_dim=64
    ):

        super().__init__()

        # CNN feature encoder
        self.encoder = CNNEncoder(
            embedding_dim=embedding_dim
        )

        # Projection head
        self.projector = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=projection_dim
        )

    def forward(self, x):

        # Extract signal features
        embedding = self.encoder(x)

        # Project features for contrastive learning
        projection = self.projector(
            embedding
        )

        return embedding, projection


def nt_xent_loss(
    z1,
    z2,
    temperature=0.5
):
    """
    NT-Xent / InfoNCE contrastive loss.

    z1:
        Projection vectors from view 1.

    z2:
        Projection vectors from view 2.

    temperature:
        Controls similarity scaling.
    """

    # Normalize projections
    z1 = F.normalize(
        z1,
        dim=1
    )

    z2 = F.normalize(
        z2,
        dim=1
    )

    batch_size = z1.size(0)

    # Combine both views
    representations = torch.cat(
        [z1, z2],
        dim=0
    )

    # Calculate similarity matrix
    similarity_matrix = torch.matmul(
        representations,
        representations.T
    )

    # Scale similarity
    similarity_matrix = (
        similarity_matrix / temperature
    )

    # Remove self-similarity
    mask = torch.eye(
        2 * batch_size,
        device=z1.device
    ).bool()

    similarity_matrix = (
        similarity_matrix.masked_fill(
            mask,
            -9e15
        )
    )

    # Positive pair indices
    positive_indices = torch.arange(
        batch_size,
        device=z1.device
    )

    positive_indices = torch.cat(
        [
            positive_indices + batch_size,
            positive_indices
        ]
    )

    # Cross entropy
    loss = F.cross_entropy(
        similarity_matrix,
        positive_indices
    )

    return loss


if __name__ == "__main__":

    print("===================================")
    print("     CONTRASTIVE MODEL TEST")
    print("===================================")

    # Batch size
    batch_size = 8

    # Create two augmented views
    # of the same type of signals
    view_1 = torch.randn(
        batch_size,
        2,
        261
    )

    view_2 = torch.randn(
        batch_size,
        2,
        261
    )

    print("\nView 1 shape:")
    print(view_1.shape)

    print("\nView 2 shape:")
    print(view_2.shape)

    # Create model
    model = ContrastiveModel(
        embedding_dim=128,
        projection_dim=64
    )

    # Process first view
    embedding_1, projection_1 = model(
        view_1
    )

    # Process second view
    embedding_2, projection_2 = model(
        view_2
    )

    print("\nEmbedding 1 shape:")
    print(embedding_1.shape)

    print("\nEmbedding 2 shape:")
    print(embedding_2.shape)

    print("\nProjection 1 shape:")
    print(projection_1.shape)

    print("\nProjection 2 shape:")
    print(projection_2.shape)

    # Calculate contrastive loss
    loss = nt_xent_loss(
        projection_1,
        projection_2
    )

    print("\nContrastive loss:")
    print(loss.item())

    print("\n===================================")
    print("   CONTRASTIVE MODEL SUCCESSFUL!")
    print("===================================")