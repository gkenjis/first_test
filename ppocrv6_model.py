"""Standalone PyTorch model for PP-OCRv6 text recognition."""
from __future__ import annotations

from typing import Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F

num_classes = 18710
config_small = {
    "model_type": "pp_ocrv6_small_rec",
    "backbone_config": {
        "model_type": "pp_lcnet_v4",
        "stem_channels": [3, 48, 96],
        "stem_type": "large",
        "block_configs": [
            [[3, 96, 96, 1, True]],
            [
                [3, 96, 96, 1, False],
                [3, 96, 96, 1, False]
            ],
            [
                [3, 96, 192, [2, 1], False],
                [3, 192, 192, 1, True],
                [3, 192, 192, 1, False],
                [3, 192, 192, 1, True],
                [3, 192, 192, 1, False],
                [3, 192, 192, 1, True],
                [3, 192, 192, 1, False]
            ],
            [
                [3, 192, 384, [2, 1], False],
                [3, 384, 384, 1, True],
                [3, 384, 384, 1, False]
            ]
        ]
    },
    "hidden_act": "silu",
    "hidden_size": 120,
    "mlp_ratio": 2.0,
    "head_out_channels": num_classes,
}

config_medium = {
    "model_type": "pp_ocrv6_small_rec",
    "backbone_config": {
        "model_type": "pp_lcnet_v4",
        "stem_channels": [3, 64, 128],
        "stem_type": "large",
        "block_configs": [
            [[3, 128, 128, 1, True]],
            [
                [3, 128, 256, 1, False],
                [3, 256, 256, 1, False],
                [3, 256, 256, 1, True]
            ],
            [
                [3, 256, 512, [2, 1], False],
                [3, 512, 512, 1, True],
                [3, 512, 512, 1, False],
                [3, 512, 512, 1, True],
                [3, 512, 512, 1, False],
                [3, 512, 512, 1, True],
                [3, 512, 512, 1, False]
            ],
            [
                [3, 512, 768, [2, 1], False],
                [3, 768, 768, 1, True],
                [3, 768, 768, 1, False]
            ]
        ]
    },
    "hidden_act": "silu",
    "hidden_size": 192,
    "mlp_ratio": 4.0,
    "depth": 2,
    "head_out_channels": 18710,
    "conv_kernel_size": [1, 7]
}

def _activation(name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU,
        "hardsigmoid": nn.Hardsigmoid,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


# class PPLCNetV4LearnableAffineBlock(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.scale = nn.Parameter(torch.tensor([1.0]))
#         self.bias = nn.Parameter(torch.tensor([0.0]))

#     def forward(self, hidden_state: Tensor) -> Tensor:
#         return self.scale * hidden_state + self.bias

class PPLCNetV4ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | list[int] = 1,
        groups: int = 1,
        activation: str | None = "relu",
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = _activation(activation) if activation else nn.Identity()

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class PPLCNetV4SqueezeExcitationModule(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
                nn.Hardsigmoid(),
            ]
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        attention = self.avg_pool(hidden_state)
        for layer in self.convolutions:
            attention = layer(attention)
        return hidden_state * attention


class PPLCNetV4LargeStem(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        channels = config["stem_channels"]
        strides = config.get("stem_strides", (2, 1, 1, 2, 1))
        activation = config.get("hidden_act", "relu")

        self.stem1 = PPLCNetV4ConvLayer(channels[0], channels[1], 3, strides[0], activation=activation)
        self.stem2a = PPLCNetV4ConvLayer(channels[1], channels[1] // 2, 2, strides[1], activation=activation)
        self.stem2b = PPLCNetV4ConvLayer(channels[1] // 2, channels[1], 2, strides[2], activation=activation)
        self.stem3 = PPLCNetV4ConvLayer(channels[1] * 2, channels[1], 3, strides[3], activation=activation)
        self.stem4 = PPLCNetV4ConvLayer(channels[1], channels[2], 1, strides[4], activation=activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        self.num_channels = config.get("num_channels", channels[0])

    def forward(self, pixel_values: Tensor) -> Tensor:
        embedding = self.stem1(pixel_values)
        embedding = F.pad(embedding, (0, 1, 0, 1))
        stem_2a = self.stem2a(embedding)
        stem_2a = F.pad(stem_2a, (0, 1, 0, 1))
        stem_2a = self.stem2b(stem_2a)
        pooled_embedding = self.pool(embedding)
        embedding = torch.cat([pooled_embedding, stem_2a], dim=1)
        embedding = self.stem3(embedding)
        return self.stem4(embedding)


class PPLCNetV4SmallStem(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        channels = config["stem_channels"]
        self.conv1 = PPLCNetV4ConvLayer(channels[0], channels[1], 3, stride=2, activation=None)
        self.act_fn = nn.GELU()
        self.conv2 = PPLCNetV4ConvLayer(channels[1], channels[2], 3, stride=2, activation=None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.conv2(self.act_fn(self.conv1(hidden_states)))


class PPLCNetV4DepthwiseSeparableConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | list[int],
        kernel_size: int,
        use_squeeze_excitation: bool,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.has_residual = in_channels == out_channels and stride == 1
        use_rep_dw = stride == 1 and in_channels == out_channels

        if use_rep_dw:
            self.token_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
            )
        else:
            self.token_conv = PPLCNetV4ConvLayer(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                groups=in_channels,
                activation=None,
            )

        self.token_squeeze_excitation = (
            PPLCNetV4SqueezeExcitationModule(in_channels, config.get("reduction", 4))
            if use_squeeze_excitation
            else nn.Identity()
        )
        self.channel_conv1 = PPLCNetV4ConvLayer(in_channels, in_channels * 2, 1, activation=None)
        self.channel_act_fn = nn.GELU()
        self.channel_conv2 = PPLCNetV4ConvLayer(in_channels * 2, out_channels, 1, activation=None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.token_conv(hidden_states)
        hidden_states = self.token_squeeze_excitation(hidden_states)
        residual = hidden_states
        hidden_states = self.channel_conv1(hidden_states)
        hidden_states = self.channel_act_fn(hidden_states)
        hidden_states = self.channel_conv2(hidden_states)
        return residual + hidden_states if self.has_residual else hidden_states


class PPLCNetV4Block(nn.Module):
    def __init__(self, config: dict[str, Any], stage_index: int) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                PPLCNetV4DepthwiseSeparableConvLayer(
                    in_channels=block[1],
                    out_channels=block[2],
                    stride=block[3],
                    kernel_size=block[0],
                    use_squeeze_excitation=block[4],
                    config=config,
                )
                for block in config["block_configs"][stage_index]
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class PPLCNetV4Encoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.convolution = PPLCNetV4LargeStem(config)
        self.blocks = nn.ModuleList([PPLCNetV4Block(config, index) for index in range(len(config["block_configs"]))])

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_state = self.convolution(pixel_values)
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return hidden_state


class PPLCNetV4Backbone(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = PPLCNetV4Encoder(config)

    def forward(self, pixel_values: Tensor) -> Tensor:
        x = self.encoder(pixel_values)
        return x


class PPOCRV6SmallRecConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | list[int] = (3, 3),
        stride: int = 1,
        activation: str | None = "silu",
        groups: int = 1,
    ) -> None:
        super().__init__()
        kernel_size = tuple(kernel_size)
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=False,
            groups=groups,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = _activation(activation) if activation else nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.activation(self.normalization(self.convolution(hidden_states)))


class PPOCRV6SmallRecAttention(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.embed_dim = config["hidden_size"]
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.scale = self.head_dim**-0.5
        self.attention_dropout = float(config.get("attention_dropout", 0.0))
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        if config.get("qkv_bias", True):
            bias = torch.cat((torch.zeros(self.embed_dim), torch.zeros(self.embed_dim), torch.zeros(self.embed_dim),))
            self.qkv.bias = nn.Parameter(bias)
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, embed_dim = hidden_states.shape
        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = mixed_qkv.reshape(
            batch_size, sequence_length, 3, self.num_heads, embed_dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = mixed_qkv.unbind(0)

        attention_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.attention_dropout, training=self.training,)
        hidden_states = torch.matmul(attention_weights, value)
        hidden_states = hidden_states.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, -1)
        return self.projection(hidden_states)


class PPOCRV6SmallRecMLP(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = int(hidden_size * config.get("mlp_ratio", 4.0))
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = _activation(config.get("hidden_act", "silu"))
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.drop = nn.Dropout(0.0)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.drop(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return self.drop(hidden_state)


class PPOCRV6SmallRecBlock(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        self.embed_dim = hidden_size
        self.self_attn = PPOCRV6SmallRecAttention(config)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps= 1e-6)
        self.mlp = PPOCRV6SmallRecMLP(config)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        return residual + self.mlp(hidden_states)


class PPOCRV6SmallRecEncoderWithSVTR(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        in_channels = config["backbone_config"]["block_configs"][-1][-1][2]
        hidden_size = config["hidden_size"]
        self.conv_block = nn.ModuleList(
            [
                PPOCRV6SmallRecConvLayer(in_channels, hidden_size, (1, 1), activation=config["hidden_act"]),
                PPOCRV6SmallRecConvLayer(in_channels, hidden_size, (1, 1), activation=config["hidden_act"]),
                PPOCRV6SmallRecConvLayer(hidden_size, hidden_size, (1, 7), activation=config["hidden_act"], groups=hidden_size),
            ]
        )
        self.svtr_block = nn.ModuleList([PPOCRV6SmallRecBlock(config) for _ in range(2)])
        self.norm = nn.LayerNorm(hidden_size, eps=config.get("layer_norm_eps", 1e-6))

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = self.conv_block[0](hidden_states)
        hidden_states = self.conv_block[1](hidden_states)
        hidden_states = hidden_states + self.conv_block[2](hidden_states)
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        for block in self.svtr_block:
            hidden_states = block(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        hidden_states = hidden_states + residual
        return hidden_states.squeeze(2).transpose(1, 2)


class PPOCRV6SmallRecHead(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = PPOCRV6SmallRecEncoderWithSVTR(config)
        self.head = nn.Linear(config["hidden_size"], config["head_out_channels"])

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.encoder(hidden_states)
        return F.softmax(self.head(hidden_states), dim=2, dtype=torch.float32).to(hidden_states.dtype)


class PPOCRV6SmallRecModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.backbone = PPLCNetV4Backbone(config["backbone_config"])

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_state = self.backbone(pixel_values)
        return hidden_state

class PPOCRV6Small(nn.Module):
    """Pure ``torch.nn`` recognition model; no Transformers dependency."""

    def __init__(self, config: dict[str, Any],weight_path) -> None:
        super().__init__()
        self.config = config
        self.model = PPOCRV6SmallRecModel(config)
        self.head = PPOCRV6SmallRecHead(config)

        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor]:
        hidden_states = self.model(pixel_values)
        hidden_states = F.avg_pool2d(hidden_states, (3, 2))
        return self.head(hidden_states)

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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_classes = logits.shape
        targets = targets.to(device=logits.device, dtype=torch.long)
        target_lengths = target_lengths.to(device="cpu", dtype=torch.long)
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
        log_probs = logits.float().log_softmax(dim=2).permute(1, 0, 2).contiguous()
        return self.loss_func(log_probs, targets, input_lengths, target_lengths)


def print_parameter_count(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'    Total: {total_num/1000000:.4f} M, Trainable: {trainable_num/1000000:.4f} M')


if __name__ == "__main__":
    IMG_H   = 48

    criterion = SimpleCTCLoss(blank_index=0)
    model = PPOCRV6Small(config_small, "ppocrv6_small.pt")

    print_parameter_count(model)

    images = torch.randn(4, 3, IMG_H, 768)
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
