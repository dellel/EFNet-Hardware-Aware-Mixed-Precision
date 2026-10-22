import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


class SpatialEncoder(nn.Module):
    """
    MobileNetV3-Small grayscale qui CONSERVE la carte spatiale.

    Au lieu d'un AdaptiveAvgPool2d(1), on réduit les canaux (576 -> D),
    puis on agrège avec une attention spatiale apprise (+ moyenne globale).
    Le modèle peut ainsi se concentrer sur le ventricule gauche.
    """

    def __init__(self, pretrained=True, temporal_dim=256):
        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        backbone = mobilenet_v3_small(weights=weights)

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

        # Réduction de canaux, sans perdre la dimension spatiale.
        self.reduce = nn.Sequential(
            nn.Conv2d(576, temporal_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, temporal_dim),
            nn.ReLU(inplace=True),
        )

        # Attention spatiale : un score par position (h, w).
        self.spatial_score = nn.Conv2d(temporal_dim, 1, kernel_size=1)

        # Fusion attention-pooling + moyenne globale.
        self.projection = nn.Linear(2 * temporal_dim, temporal_dim)

    def forward(self, x):
        # x: (B*T, 1, H, W)
        x = self.features(x)          # (B*T, 576, h, w)
        x = self.reduce(x)            # (B*T, D, h, w)

        weights = torch.softmax(
            self.spatial_score(x).flatten(2), dim=-1
        )                             # (B*T, 1, h*w)
        attended = (x.flatten(2) * weights).sum(dim=-1)   # (B*T, D)
        average = x.mean(dim=(2, 3))                      # (B*T, D)

        return self.projection(torch.cat([attended, average], dim=1))


class MultiScaleDepthwiseTemporalBlock(nn.Module):
    """
    Bloc temporel multiscalaire léger, empilable.
    Entrée et sortie : (B, C, T)
    Deux branches depthwise : dilatation d et 2d.
    """

    def __init__(self, dim=256, dilation=1, dropout=0.1):
        super().__init__()

        self.local = nn.Conv1d(
            dim, dim, kernel_size=3,
            padding=dilation, dilation=dilation,
            groups=dim, bias=False,
        )
        self.dilated = nn.Conv1d(
            dim, dim, kernel_size=3,
            padding=2 * dilation, dilation=2 * dilation,
            groups=dim, bias=False,
        )
        self.fusion = nn.Conv1d(2 * dim, dim, kernel_size=1, bias=False)

        # GroupNorm : indépendant de la taille de batch (batch = 2 ici).
        self.norm = nn.GroupNorm(8, dim)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        combined = torch.cat([self.local(x), self.dilated(x)], dim=1)
        combined = self.drop(self.norm(self.fusion(combined)))
        return self.act(combined + x)


class TemporalAttentionPooling(nn.Module):
    """Agrégation temporelle pondérée et apprise."""

    def __init__(self, dim=256):
        super().__init__()
        self.score = nn.Conv1d(dim, 1, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B, C, T)
        weights = torch.softmax(self.score(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


class LiteMSTemporalEFNet(nn.Module):
    """
    v2 : encodeur spatial avec attention, pile de blocs temporels
    (dilatations 1, 2, 4, 8 -> champ réceptif ~61 images),
    agrégation temporelle apprise, régression de la LVEF (sortie dans [0, 1]).
    """

    def __init__(
        self,
        seq_len=64,
        temporal_dim=256,
        pretrained=True,
        dropout=0.2,
        dilations=(1, 2, 4, 8),
        freeze_backbone_bn=True,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.freeze_backbone_bn = freeze_backbone_bn

        self.encoder = SpatialEncoder(
            pretrained=pretrained,
            temporal_dim=temporal_dim,
        )

        self.temporal = nn.Sequential(*[
            MultiScaleDepthwiseTemporalBlock(
                dim=temporal_dim, dilation=d, dropout=0.1
            )
            for d in dilations
        ])

        self.temporal_pool = TemporalAttentionPooling(dim=temporal_dim)

        self.regressor = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    # -- utilitaires d'entraînement ---------------------------------

    def set_backbone_frozen(self, frozen: bool):
        """Gèle / dégèle les poids du backbone MobileNet."""
        for p in self.encoder.features.parameters():
            p.requires_grad = not frozen

    def train(self, mode=True):
        super().train(mode)
        # Avec ~128 images corrélées par batch, les statistiques BatchNorm
        # du backbone sont bruitées : on les garde figées (mode eval).
        if mode and self.freeze_backbone_bn:
            for m in self.encoder.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    # -----------------------------------------------------------------

    def forward(self, x):
        # x: (B, T, 1, H, W)
        B, T, C, H, W = x.shape

        features = self.encoder(x.reshape(B * T, C, H, W))   # (B*T, D)
        features = features.reshape(B, T, -1).transpose(1, 2)  # (B, D, T)

        features = self.temporal(features)
        features = self.temporal_pool(features)

        return self.regressor(features)


if __name__ == "__main__":
    model = LiteMSTemporalEFNet(seq_len=64, pretrained=False)
    model.eval()

    x = torch.randn(1, 64, 1, 224, 224)
    with torch.no_grad():
        y = model(x)

    total = sum(p.numel() for p in model.parameters())
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Total parameters:", total)
