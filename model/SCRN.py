# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_, DropPath


class WMSA(nn.Module):
    """
    Window Multi-head Self Attention (W-MSA / SW-MSA).
    Используется в Swin Transformer для обработки локальных окон.
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, attn_type):
        """
        Args:
            input_dim (int): размерность входа (число каналов)
            output_dim (int): размерность выхода
            head_dim (int): размерность одного attention head
            window_size (int): размер окна
            attn_type (str): 'W' (window) или 'SW' (shifted window)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = attn_type

        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )
        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = nn.Parameter(
            self.relative_position_params
            .view(2 * window_size - 1, 2 * window_size - 1, self.n_heads)
            .permute(2, 0, 1)
        )

        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def generate_mask(self, h, w, p, shift):
        """
        Генерация маски для SW-MSA (shifted window).
        Маска запрещает внимание за пределы сдвинутого окна.
        """
        attn_mask = torch.zeros(
            h, w, p, p, p, p, dtype=torch.bool,
            device=self.relative_position_params.device
        )

        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True

        attn_mask = rearrange(
            attn_mask,
            "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)"
        )
        return attn_mask

    def forward(self, x):
        """
        Forward pass:
        - (опционально) сдвигаем изображение
        - разбиваем на окна
        - считаем self-attention внутри каждого окна
        - восстанавливаем изображение
        """
        if self.type != "W":
            x = torch.roll(
                x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2)
            )

        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size, p2=self.window_size
        )
        h_windows, w_windows = x.size(1), x.size(2)

        x = rearrange(x, "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c")
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim
        ).chunk(3, dim=0)

        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        sim += rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")

        if self.type != "W":
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)

        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")

        output = self.linear(output)

        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows, p1=self.window_size
        )

        if self.type != "W":
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))

        return output

    def relative_embedding(self):
        """Генерация относительных позиционных эмбеддингов для каждого окна."""
        cord = torch.tensor(
            np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)])
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[
            :, relation[:, :, 0].long(), relation[:, :, 1].long()
        ]


class Block(nn.Module):
    """Swin Transformer Block: LayerNorm → WMSA → MLP."""

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, attn_type="W", input_resolution=None):
        super().__init__()
        assert attn_type in ["W", "SW"]

        if input_resolution <= window_size:
            attn_type = "W"

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, attn_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    """
    Комбинация сверточного блока + Swin Transformer блока.
    Позволяет объединить локальные свертки и глобальные зависимости.
    """

    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path=0.1, attn_type="W", input_resolution=None):
        super().__init__()

        if input_resolution <= window_size:
            attn_type = "W"

        self.trans_block = Block(trans_dim, trans_dim, head_dim, window_size, drop_path, attn_type, input_resolution)

        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, kernel_size=1, stride=1, bias=True)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, kernel_size=1, stride=1, bias=True)

        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
        )

        self.conv_dim = conv_dim
        self.trans_dim = trans_dim

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)

        conv_x = self.conv_block(conv_x) + conv_x

        trans_x = Rearrange("b c h w -> b h w c")(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange("b h w c -> b c h w")(trans_x)

        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x


class SCRN(nn.Module):
    """
    Swin-Convolutional Residual Network (SCRN).
    Сеть объединяет Conv-блоки и Swin Transformer для обработки изображений.
    """

    def __init__(self, in_nc=1, config=[1, 1, 1, 1, 1], dim=64, drop_path_rate=0.0, input_resolution=128):
        super().__init__()
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        # Слои сети
        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))

        begin = 0
        self.m1 = nn.Sequential(
            *[ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                             "W" if not i % 2 else "SW", input_resolution) for i in range(config[0])],
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        )

        begin += config[0]
        self.m2 = nn.Sequential(
            *[ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                             "W" if not i % 2 else "SW", input_resolution) for i in range(config[1])],
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        )

        begin += config[1]
        self.m3 = nn.Sequential(
            *[ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                             "W" if not i % 2 else "SW", input_resolution) for i in range(config[2])],
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        )

        begin += config[2]
        self.m4 = nn.Sequential(
            *[ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                             "W" if not i % 2 else "SW", input_resolution) for i in range(config[3])],
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        )

        begin += config[3]
        self.m5 = nn.Sequential(
            *[ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                             "W" if not i % 2 else "SW", input_resolution) for i in range(config[4])],
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        )

        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))

        self.apply(self._init_weights)

    def forward(self, x0):
        """Прямой проход сети."""
        h, w = x0.size()[-2:]

        x1 = self.m_head(x0)
        x2 = self.m1(x1)
        x3 = self.m2(x2)
        x4 = self.m3(x3)
        x5 = self.m4(x4 + x3)
        x6 = self.m5(x5 + x2)
        x7 = self.m_tail(x6 + x1)

        x7 = x7[..., :h, :w]
        return x7

    @staticmethod
    def _init_weights(m):
        """Инициализация весов в стиле Swin Transformer."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == "__main__":
    net = SCRN()
    print(net)

    x = torch.randn((1, 1, 128, 128))
    y = net(x)
    print("Output shape:", y.shape)
