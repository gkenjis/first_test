"""Standalone PyTorch model for PP-OCRv6 text recognition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class PPLCNetV4LearnableAffineBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.0]))
        self.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.scale * hidden_state + self.bias


class PPLCNetV4ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | list[int] = 1,
        groups: int = 1,
        activation: str | None = "relu",
        use_learnable_affine_block: bool = False,
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
        self.lab = PPLCNetV4LearnableAffineBlock() if activation and use_learnable_affine_block else nn.Identity()

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return self.lab(hidden_state)


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
        use_lab = config.get("use_learnable_affine_block", False)
        activation = config.get("hidden_act", "relu")

        self.stem1 = PPLCNetV4ConvLayer(
            channels[0], channels[1], 3, strides[0], activation=activation,
            use_learnable_affine_block=use_lab,
        )
        self.stem2a = PPLCNetV4ConvLayer(
            channels[1], channels[1] // 2, 2, strides[1], activation=activation,
            use_learnable_affine_block=use_lab,
        )
        self.stem2b = PPLCNetV4ConvLayer(
            channels[1] // 2, channels[1], 2, strides[2], activation=activation,
            use_learnable_affine_block=use_lab,
        )
        self.stem3 = PPLCNetV4ConvLayer(
            channels[1] * 2, channels[1], 3, strides[3], activation=activation,
            use_learnable_affine_block=use_lab,
        )
        self.stem4 = PPLCNetV4ConvLayer(
            channels[1], channels[2], 1, strides[4], activation=activation,
            use_learnable_affine_block=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        self.num_channels = config.get("num_channels", channels[0])

    def forward(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.shape[1] != self.num_channels:
            raise ValueError("Input channel count does not match the model configuration")

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
        self.convolution = (
            PPLCNetV4LargeStem(config)
            if config.get("stem_type", "large") == "large"
            else PPLCNetV4SmallStem(config)
        )
        self.blocks = nn.ModuleList(
            [PPLCNetV4Block(config, index) for index in range(len(config["block_configs"]))]
        )

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
        return self.encoder(pixel_values)


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
        self.num_heads = config.get("num_attention_heads", 8)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.scale = self.head_dim**-0.5
        self.attention_dropout = float(config.get("attention_dropout", 0.0))
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        if config.get("qkv_bias", True):
            bias = torch.cat(
                (
                    torch.zeros(self.embed_dim),
                    torch.zeros(self.embed_dim),
                    torch.zeros(self.embed_dim),
                )
            )
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
        attention_weights = F.dropout(
            attention_weights,
            p=self.attention_dropout,
            training=self.training,
        )
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
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=config.get("layer_norm_eps", 1e-6))
        self.mlp = PPOCRV6SmallRecMLP(config)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=config.get("layer_norm_eps", 1e-6))

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
                PPOCRV6SmallRecConvLayer(
                    hidden_size,
                    hidden_size,
                    config.get("conv_kernel_size", [1, 7]),
                    activation=config["hidden_act"],
                    groups=hidden_size,
                ),
            ]
        )
        self.svtr_block = nn.ModuleList(
            [PPOCRV6SmallRecBlock(config) for _ in range(config.get("depth", 2))]
        )
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
        return F.avg_pool2d(hidden_state, (3, 2))

class PPOCRV6Small(nn.Module):
    """Pure ``torch.nn`` recognition model; no Transformers dependency."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.model = PPOCRV6SmallRecModel(config)
        self.head = PPOCRV6SmallRecHead(config)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor]:
        hidden_states = self.model(pixel_values)
        return (self.head(hidden_states),)

    @classmethod
    def from_files(
        cls,
        config_path: str | Path,
        weights_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> PPOCRV6Small:
        with Path(config_path).open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        model = cls(config)
        state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict, strict=strict)
        return model