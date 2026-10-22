import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


class SpatialEncoder(nn.Module):
    """MobileNetV3-Small adapté aux images échocardiographiques grayscale."""

    def __init__(self, pretrained=True, temporal_dim=128):
        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = mobilenet_v3_small(weights=weights)

        # Adaptation de la première convolution RGB vers grayscale.
        old_conv = backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )

        backbone.features[0][0] = new_conv

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Linear(
            576,
            temporal_dim,
            bias=True,
        )

    def forward(self, x):
        # x: (B*T, 1, H, W)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)

        return x


class MultiScaleDepthwiseTemporalBlock(nn.Module):
    """
    Bloc temporel multiscalaire léger.

    Entrée et sortie : (B, C, T)
    """

    def __init__(self, dim=128):
        super().__init__()

        # Branche temporelle locale.
        self.local = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )

        # Branche temporelle à plus grande portée.
        self.dilated = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=dim,
            bias=False,
        )

        # Fusion des deux échelles temporelles.
        self.fusion = nn.Conv1d(
            2 * dim,
            dim,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        local_features = self.local(x)
        dilated_features = self.dilated(x)

        combined = torch.cat(
            [local_features, dilated_features],
            dim=1,
        )

        combined = self.fusion(combined)
        combined = self.bn(combined)

        return self.act(combined + identity)


class TemporalAttentionPooling(nn.Module):
    """Agrégation temporelle pondérée et apprise."""

    def __init__(self, dim=128):
        super().__init__()

        self.score = nn.Conv1d(
            dim,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        # x: (B, C, T)
        weights = torch.softmax(
            self.score(x),
            dim=-1,
        )

        pooled = torch.sum(
            x * weights,
            dim=-1,
        )

        return pooled


class LiteMSTemporalEFNet(nn.Module):
    """
    Encodeur spatial MobileNetV3-Small,
    bloc temporel multiscalaire léger,
    agrégation temporelle apprise,
    régression de la LVEF.
    """

    def __init__(
        self,
        seq_len=32,
        temporal_dim=128,
        pretrained=True,
        dropout=0.2,
    ):
        super().__init__()

        self.seq_len = seq_len

        self.encoder = SpatialEncoder(
            pretrained=pretrained,
            temporal_dim=temporal_dim,
        )

        self.temporal = MultiScaleDepthwiseTemporalBlock(
            dim=temporal_dim,
        )

        self.temporal_pool = TemporalAttentionPooling(
            dim=temporal_dim,
        )

        self.regressor = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, T, 1, H, W)
        B, T, C, H, W = x.shape

        # Extraction spatiale avec poids partagés.
        x = x.reshape(B * T, C, H, W)
        features = self.encoder(x)

        # (B*T, D) -> (B, D, T)
        features = features.reshape(
            B, T, -1
        ).transpose(1, 2)

        # Modélisation temporelle.
        features = self.temporal(features)

        # Agrégation temporelle pondérée.
        features = self.temporal_pool(features)

        # Régression.
        prediction = self.regressor(features)

        return prediction


if __name__ == "__main__":
    model = LiteMSTemporalEFNet(
        seq_len=32,
        temporal_dim=128,
        pretrained=False,
    )

    model.eval()

    x = torch.randn(1, 32, 1, 224, 224)

    with torch.no_grad():
        y = model(x)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
