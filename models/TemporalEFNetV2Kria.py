import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


class FrameEncoder(nn.Module):

  def __init__(self):
    super().__init__()

    # MODIFICATION 1 : Chargement des poids pré-entraînés ImageNet
    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    old_conv = backbone.features[0][0]

    new_conv = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    # MODIFICATION 2 : Copie et moyenne des poids RGB vers 1 canal (Grayscale)
    with torch.no_grad():
      new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

    backbone.features[0][0] = new_conv

    self.features = backbone.features
    self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
    self.proj = nn.Conv2d(576, 576, kernel_size=1, bias=True)

  def forward(self, x):
    x = self.features(x)
    x = self.pool(x)
    x = self.proj(x)
    return x


class FPGA_TemporalBlock(nn.Module):

  def __init__(self, dim=576, seq_len=32):
    super().__init__()

    self.seq_len = seq_len

    # MODIFICATION 3 : Bloc temporel amélioré avec connexion résiduelle (Residual)
    # Branche principale (Feature extraction temporelle plus riche)
    self.depthwise1 = nn.Conv2d(
        dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False
    )
    self.bn1 = nn.BatchNorm2d(dim)

    self.depthwise2 = nn.Conv2d(
        dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False
    )
    self.bn2 = nn.BatchNorm2d(dim)

    self.pointwise = nn.Conv2d(dim, 256, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(256)

    self.act = nn.ReLU(inplace=True)

    self.pool_temp = nn.AvgPool2d(kernel_size=(seq_len, 1))

  def forward(self, x):
    # Sauvegarde pour le résidu (shortcut)
    identity = x

    # Première convolution depthwise
    out = self.depthwise1(x)
    out = self.bn1(out)
    out = self.act(out)

    # Seconde convolution depthwise pour élargir le champ récepteur
    out = self.depthwise2(out)
    out = self.bn2(out)

    # Connexion résiduelle (addition sur le même nombre de canaux 'dim')
    out = self.act(out + identity)

    # Pointwise pour réduire la dimension vers 256
    out = self.pointwise(out)
    out = self.bn3(out)
    out = self.act(out)

    # Pooling temporel global sur la dimension de la séquence
    out = self.pool_temp(out)

    out = torch.flatten(out, 1)

    return out


class Regressor(nn.Module):

  def __init__(self):
    super().__init__()

    self.mlp = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, 1),
    )

  def forward(self, x):
    return self.mlp(x)


class TemporalEFNetV2Kria(nn.Module):

  def __init__(self, seq_len=32):
    super().__init__()

    self.seq_len = seq_len

    self.encoder = FrameEncoder()
    self.temporal = FPGA_TemporalBlock(dim=576, seq_len=seq_len)
    self.regressor = Regressor()

  def forward(self, x):
    B, T, C, H, W = x.shape

    x = x.view(B * T, C, H, W)

    feats = self.encoder(x)

    feats = feats.view(B, T, 576)

    feats = feats.permute(0, 2, 1)

    feats = feats.unsqueeze(3)

    context = self.temporal(feats)

    out = self.regressor(context)

    return out


if __name__ == "__main__":

  weights_path = "/workspace/echonet_int8/TemporalEFNetV2Kria_state_dict.pth"

  print("Creating model...")

  model = TemporalEFNetV2Kria(seq_len=32)

  print("Loading state_dict...")

  # Note : Comme la première couche et le bloc temporel ont changé,
  # un chargement strict (strict=True) plantera si les dimensions des poids sauvegardés
  # ne correspondent plus. Si vous partez de nouveaux poids pré-entraînés,
  # pensez à réentraîner le modèle.
  try:
    state_dict = torch.load(weights_path, map_location="cpu")
    result = model.load_state_dict(state_dict, strict=False)
    print("State_dict loaded with strict=False.")
    print("Missing keys   :", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)
  except Exception as e:
    print(
        "Impossible de charger l'ancien state_dict (ce qui est normal suite aux"
        f" modifications architecturales) : {e}"
    )
    print(
        "Le modèle est initialisé pour un nouveau cycle de fine-tuning à partir"
        " de ImageNet."
    )

  model.eval()

  print("Parameters:", sum(p.numel() for p in model.parameters()))

  dummy = torch.randn(1, 32, 1, 224, 224)

  with torch.no_grad():
    output = model(dummy)

  print("Input shape :", tuple(dummy.shape))
  print("Output shape:", tuple(output.shape))
  print("Output:", output)