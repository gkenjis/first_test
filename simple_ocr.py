import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class ConformerFeedForward(nn.Module):
    """Pre-norm position-wise feed-forward module used by a Conformer block."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ConformerSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention module for [B, T, C] tensors."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        normalized = self.norm(x)
        output, _ = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=False)
        return self.dropout(output)


class ConformerConvolution(nn.Module):
    """Conformer convolution module for [B, T, C] sequence features."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1, use_group_norm: bool = True) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model, bias=True)
        self.channel_norm = nn.GroupNorm(1, d_model) if use_group_norm else nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x).transpose(1, 2).contiguous()
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.channel_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2).contiguous()


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block preserving shape [B, T, C]."""

    def __init__(self, d_model: int = 256, nhead: int = 4, dim_feedforward: int = 1024, kernel_size: int = 15, dropout: float = 0.1, use_group_norm: bool = True, convolution_first: bool = False) -> None:
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.self_attention = ConformerSelfAttention(d_model, nhead, dropout)
        self.convolution = ConformerConvolution(d_model, kernel_size, dropout, use_group_norm)
        self.ffn2 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.convolution_first = convolution_first

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        x = x + 0.5 * self.ffn1(x)
        if self.convolution_first:
            x = x + self.convolution(x)
        x = x + self.self_attention(x, padding_mask)
        if not self.convolution_first:
            x = x + self.convolution(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks for sequence features shaped [B, T, C]."""

    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 4, dim_feedforward: int = 1024, kernel_size: int = 15, dropout: float = 0.1, use_group_norm: bool = True, convolution_first: bool = False) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")
        self.layers = nn.ModuleList([ConformerBlock(d_model, nhead, dim_feedforward, kernel_size, dropout, use_group_norm, convolution_first) for _ in range(num_layers)])

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, T, C], got {tuple(x.shape)}")
        if padding_mask is not None and padding_mask.shape != x.shape[:2]:
            raise ValueError(f"padding_mask must have shape {tuple(x.shape[:2])}, got {tuple(padding_mask.shape)}")
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x

class PositionalEncoding(nn.Module):
    """可変長系列に対応するSin/Cos位置エンコーディング。"""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()

        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError("d_modelは正の偶数である必要があります。")

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.size(2) != self.d_model:
            raise ValueError(f"xは[B, T, {self.d_model}]である必要があります: shape={tuple(x.shape)}")

        position = torch.arange(x.size(1), device=x.device, dtype=torch.float32).unsqueeze(1)
        frequency = torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32)
        divisor = torch.exp(frequency * (-math.log(10000.0) / self.d_model))
        encoding = torch.zeros((1, x.size(1), self.d_model), device=x.device, dtype=torch.float32)
        encoding[0, :, 0::2] = torch.sin(position * divisor)
        encoding[0, :, 1::2] = torch.cos(position * divisor)

        return self.dropout(x + encoding.to(dtype=x.dtype))


class ConvBlock(nn.Sequential):
    """Conv2d、GroupNorm、GELUによる畳み込みブロック。"""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        if out_channels % 8 != 0:
            raise ValueError(f"out_channelsは8で割り切れる必要があります: {out_channels}")

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class ResidualBlock(nn.Module):
    """チャネル数と空間サイズを維持するResidual Block。"""

    def __init__(self, channels: int) -> None:
        super().__init__()

        if channels % 8 != 0:
            raise ValueError(f"channelsは8で割り切れる必要があります: {channels}")

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )

        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


class SimpleLineOCR(nn.Module):
    """
    入力高さの比較実験に対応するラインOCRモデル。
    入力:
        [B, 3, H, W]
    出力:
        [B, T, num_classes]
    高さ:
        32、40、48、64などを使用可能。
    幅:
        可変。CNNによる横方向圧縮率は2。
    """
    def __init__(self, num_classes: int, input_channels: int = 3, d_model: int = 256) -> None:
        super().__init__() 
        if d_model % 8 != 0:
            raise ValueError("d_modelは8で割り切れる必要があります。")

        self.input_channels = input_channels

        self.cnn = nn.Sequential(
            ConvBlock(input_channels, 32, stride=(2, 2)),
            ResidualBlock(32),
            ConvBlock(32, 64, stride=(2, 1)),
            ResidualBlock(64),
            ConvBlock(64, 128, stride=(2, 2)),
            ResidualBlock(128),
            ConvBlock(128, 192, stride=(2, 1)),
            ResidualBlock(192),
            ConvBlock(192, d_model, stride=(2, 1)),
            ResidualBlock(d_model),
        )
        use_conformer = False

        self.height_pool = nn.AdaptiveAvgPool2d((1, None))
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=0.1)
        if use_conformer:
            self.sequence_model = ConformerEncoder(d_model=d_model, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.1)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=1024, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            self.sequence_model = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.cnn(images)
        features = self.height_pool(features).squeeze(2).transpose(1, 2).contiguous()
        features = self.positional_encoding(features)

        features = self.sequence_model(features)
        return self.classifier(features)
    
class SimpleCTCLoss(nn.Module):
    """
    パディング済み2次元targetsに対応するCTC Loss。
    logits:
        [B, T, C]
    targets:
        [B, S]
    target_lengths:
        [B]
    target_lengthsより後ろのtargetsはパディングとして無視される。
    blank_index=0の場合、パディング値にも0を使用できる。
    """

    def __init__(self, blank_index: int = 0) -> None:
        super().__init__()

        if blank_index < 0:
            raise ValueError("blank_indexは0以上である必要があります。")

        self.blank_index = blank_index
        self.loss_func = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)

    def forward(self, logits: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
        batch_size, time_steps, num_classes = logits.shape
        targets = targets.to(device=logits.device, dtype=torch.long)
        target_lengths = target_lengths.to(device="cpu", dtype=torch.long)
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
        log_probs = logits.float().log_softmax(dim=2).permute(1, 0, 2).contiguous()
        return self.loss_func(log_probs, targets, input_lengths, target_lengths)
    

@torch.no_grad()
def decode_ctc(logits: Tensor, tokenizer=None, blank_index: int = 0) -> list[str]:
    token_ids = logits.argmax(dim=-1)  # [B, T]
    texts = []
    for sequence in token_ids:
        sequence = torch.unique_consecutive(sequence)
        sequence = sequence[sequence != blank_index]
        texts.append(tokenizer.decode(sequence.tolist()))

    return texts

def print_parameter_count(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'    Total: {total_num/1000000:.4f} M, Trainable: {trainable_num/1000000:.4f} M')

if __name__ == "__main__":
    num_classes = 180

    model = SimpleLineOCR(num_classes=num_classes, input_channels=3, d_model=256)
    criterion = SimpleCTCLoss(blank_index=0)
    print_parameter_count(model)

    images = torch.randn(4, 3, 48, 600)
    logits = model(images)
    targets = torch.tensor([
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0],
        [7, 8, 9],
    ], dtype=torch.long)

    target_lengths = torch.tensor([3, 2, 1, 3], dtype=torch.long)
    loss = criterion(logits, targets, target_lengths)
    print(f"Input Shape(Image): {images.shape}")
    print(f"Output Shape(Logits): {logits.shape}")
    print(loss.item())
    # decode_ctc(logits)