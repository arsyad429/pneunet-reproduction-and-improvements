# %% [markdown]
# # PneuNet E7-T
# Arsitektur lengkap classifier final, dari tensor P3 hingga logit PNEUMONIA.
# Input berbentuk [B, 1, 224, 224] dalam FP32 dengan piksel 0–255.

# %% Imports dan inisialisasi
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


def _initialize(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _initialize_residual(module, seed, name):
    offset = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed((seed + offset) % 2**31)
        module.apply(_initialize)


# %% Blok ekstraksi fitur
class GrayToRGB(nn.Module):
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError('Expected NCHW grayscale input.')
        return images.repeat(1, 3, 1, 1)


class SamePadConv2d(nn.Conv2d):
    """Pad to ceil(input / stride), including the asymmetric stride-2 case."""

    def forward(self, features):
        height, width = features.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        ph = max(0, ((height - 1) // sh) * sh + dh * (kh - 1) + 1 - height)
        pw = max(0, ((width - 1) // sw) * sw + dw * (kw - 1) + 1 - width)
        padded = F.pad(features, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return F.conv2d(padded, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn, *, stride=1, dilation=1, depthwise=False):
        super().__init__()
        groups = in_channels if depthwise else 1
        self.conv = SamePadConv2d(in_channels, out_channels, kernel_size, stride=stride,
                                 dilation=dilation, groups=groups, bias=True)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, features):
        return self.relu(self.bn(self.conv(features)))


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(1, channels // 16)
        self.reduce = nn.Linear(channels, hidden)
        self.expand = nn.Linear(hidden, channels)

    def forward(self, features):
        channel_mean = features.mean(dim=(2, 3))
        gate = torch.sigmoid(self.expand(F.relu(self.reduce(channel_mean))))
        return features * gate[:, :, None, None]


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels, dropout, bn):
        super().__init__()
        self.depthwise = ConvBNReLU(channels, channels, 3, bn, depthwise=True)
        self.pointwise = ConvBNReLU(channels, channels, 1, bn)
        self.se = SEBlock(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        return self.dropout(self.se(self.pointwise(self.depthwise(features))))


class ConcatenateChannels(nn.Module):
    def forward(self, branches):
        return torch.cat(branches, dim=1)


class DetailResidualBlock(nn.Module):
    def __init__(self, channels, bn):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = bn(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = bn(channels)
        self.relu_out = nn.ReLU()

    def forward(self, features):
        first = self.relu1(self.bn1(self.conv1(features)))
        return self.relu_out(features + self.bn2(self.conv2(first)))


# %% Context refinement, GRN, dan pooling
class ContextBottleneck(nn.Module):
    def __init__(self, dilation, bn):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), bn(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=dilation, dilation=dilation, groups=64, bias=False),
            bn(64), nn.ReLU(),
            nn.Conv2d(64, 128, 1, bias=False), bn(128),
        )
        self.relu_out = nn.ReLU()

    def forward(self, features):
        return self.relu_out(features + self.spatial(features))


class GlobalResponseNorm2d(nn.Module):
    def __init__(self, channels=128, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.offset = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, features):
        energy = torch.linalg.vector_norm(features, ord=2, dim=(-2, -1), keepdim=True)
        relative = energy / (energy.mean(dim=1, keepdim=True) + self.eps)
        return features + self.scale * features * relative + self.offset


class NormalizedLearnableSpatialPooling(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.attention_hidden = nn.Conv2d(channels, channels, 1, bias=True)
        self.attention_activation = nn.ReLU()
        self.attention_conv = nn.Conv2d(channels, 1, 1, bias=True)

    def attention(self, features):
        hidden = self.attention_activation(self.attention_hidden(features))
        return torch.sigmoid(self.attention_conv(hidden))

    def forward(self, features):
        attention = self.attention(features)
        weighted = (features * attention).sum(dim=(2, 3))
        return weighted / attention.sum(dim=(2, 3)).clamp_min(1e-6)


# %% Classifier E7-T lengkap
class PneuNetE7T(nn.Module):
    """E7-T with 229,544 parameters and one binary logit per input image."""

    def __init__(self, *, seed=42, dropout=0.3, bn_eps=0.001, bn_momentum=0.01):
        super().__init__()

        def bn(channels):
            return nn.BatchNorm2d(channels, eps=bn_eps, momentum=bn_momentum)

        # Keep layer creation and initialization order for seed42 reproduction.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.gray_to_rgb = GrayToRGB()
            self.stem = ConvBNReLU(3, 32, 3, bn)
            self.pointwise_branch0 = ConvBNReLU(32, 32, 1, bn)
            self.depthwise_branches = nn.ModuleList(
                DepthwiseSeparableBlock(32, dropout, bn) for _ in range(3)
            )
            self.feature_concat = ConcatenateChannels()
            self.downsample = nn.Sequential(
                ConvBNReLU(128, 128, 3, bn, stride=2, depthwise=True), nn.Dropout(dropout)
            )
            self.aspp_1x1 = ConvBNReLU(128, 32, 1, bn)
            self.aspp_atrous = nn.ModuleList(
                ConvBNReLU(128, 32, 3, bn, dilation=dilation) for dilation in (1, 3, 6)
            )
            self.aspp_fuse = ConvBNReLU(128, 128, 1, bn)
            self.learnable_pooling = NormalizedLearnableSpatialPooling()
            self.classifier_dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(128, 1)
            self.apply(_initialize)

            self.detail_refine = nn.Sequential(*(DetailResidualBlock(32, bn) for _ in range(2)))
            _initialize_residual(self.detail_refine, seed, 'detail_refine')
            self.context_refine = nn.Sequential(*(ContextBottleneck(d, bn) for d in (1, 2)))
            _initialize_residual(self.context_refine, seed, 'narrow_context')
            self.context_refine.add_module('response_norm', GlobalResponseNorm2d())

    def forward(self, images):
        features = self.detail_refine(self.stem(self.gray_to_rgb(images)))
        branches = [self.pointwise_branch0(features)]
        branches += [branch(features) for branch in self.depthwise_branches]
        features = self.downsample(self.feature_concat(branches))
        levels = [self.aspp_1x1(features)] + [branch(features) for branch in self.aspp_atrous]
        features = self.context_refine(self.aspp_fuse(torch.cat(levels, dim=1)))
        return self.classifier(self.classifier_dropout(self.learnable_pooling(features)))


# %% Memuat checkpoint T
def build_model(config=None):
    config = {} if config is None else config
    args = {key: config[key] for key in ('seed', 'dropout', 'bn_eps', 'bn_momentum') if key in config}
    return PneuNetE7T(**args)


def load_checkpoint(path):
    """Load the original T best.pt or a best.pt from train_pneunet_e7_t.py."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'signature' in checkpoint:
        if checkpoint['signature']['candidate'] != 'T':
            raise ValueError('This loader expects a checkpoint for candidate T.')
        config = checkpoint['signature']['config']
    else:
        if checkpoint.get('model_name') != 'PneuNetE7T':
            raise ValueError('This loader expects a PneuNetE7T checkpoint.')
        config = checkpoint['config']
    model = build_model(config)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval()


# %% Cek arsitektur tanpa training
if __name__ == '__main__':
    model = build_model()
    print(f'E7-T parameters: {sum(p.numel() for p in model.parameters()):,}')
