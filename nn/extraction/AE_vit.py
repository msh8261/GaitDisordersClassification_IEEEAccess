# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103

import torch
import torch.nn as nn

import config.config_train as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "selu": nn.SELU(),
    "elu": nn.ELU(),
    "LeakyReLU": nn.LeakyReLU(0.1),
    "rrelu": nn.RReLU(0.1, 0.3),
}

input_size = config.params["input_size"]
sequence_length = config.params["sequences"]
dropout = nn.Dropout(config.params["dropout"])
activation_function = activations[config.params["acf_indx"]]
bottleneck = config.params["bottleneck"]
last_layer = config.params["last_layer"]
num_class = config.params["num_class"]


class Attention_encoder(nn.Module):
    """Attention mechanism."""

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches , 3 * dim)
        # print(qkv.shape)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches , 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches )
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches , n_patches )
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches , n_patches )
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches , head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches , n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches , dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches , dim)
        x = self.proj_drop(x)  # (n_samples, n_patches , dim)

        return x


class MLP(nn.Module):
    """Multilayer perceptron."""

    def __init__(self, in_features, hidden_features, out_features, p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)  # (n_samples, n_patches , hidden_features)
        x = self.act(x)  # (n_samples, n_patches , hidden_features)
        x = self.drop(x)  # (n_samples, n_patches , hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches , out_features)
        x = self.drop(x)  # (n_samples, n_patches , out_features)
        return x


class Attention_decoder(nn.Module):
    """Attention mechanism."""

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches , 3 * dim)
        # print(qkv.shape)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches , 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches )
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches , n_patches )
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches , n_patches )
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches , head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches , n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches , dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches , dim)
        x = self.proj_drop(x)  # (n_samples, n_patches , dim)

        return x


class Block_Encoder(nn.Module):
    """Transformer block."""

    def __init__(self, dim, n_heads, mlp_ratio=4, qkv_bias=True, p=0.0, attn_p=0.0):
        super().__init__()
        self.btn = bottleneck
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention_encoder(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Block_Decoder(nn.Module):
    """Transformer block."""

    def __init__(self, dim, n_heads, mlp_ratio=4, qkv_bias=True, p=0.0, attn_p=0.0):
        super().__init__()
        self.btn = bottleneck
        self.inverted_bottleneck_layer = nn.Linear(self.btn, input_size)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention_decoder(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        x = self.inverted_bottleneck_layer(x)
        x = x + self.norm2(self.mlp(x))
        x = x + self.norm1(self.attn(x))
        return x


class AutoEncoderViT(nn.Module):
    """Simplified implementation of the Vision transformer."""

    def __init__(
        self,
        depth=2,  # number of TF blocks, best 2
        n_heads=2,  # best 2
        mlp_ratio=2,  # best 2
        qkv_bias=True,
        p=0.1,  # 0.1 best
        attn_p=0.1,  # best 0.1
    ):
        super().__init__()
        self.reduced_features = 0  # with this reach 88%
        self.input_size = input_size - self.reduced_features
        self.device = device

        self.btn = bottleneck

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.input_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, sequence_length, self.input_size)
        ).to(device)
        self.pos_drop = nn.Dropout(p=p)
        self.tanh = nn.Tanh()

        self.blocks_encoder = nn.ModuleList(
            [
                Block_Encoder(
                    dim=self.input_size,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.bottleneck_layer = nn.Linear(self.input_size, self.btn)

        self.blocks_decoder = nn.ModuleList(
            [
                Block_Decoder(
                    dim=self.input_size,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(self.input_size, eps=1e-6)

        self.classifier = nn.Linear(self.btn, num_class)

    def forward(self, x):
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        for block_en in self.blocks_encoder:
            encoded = block_en(x)

        btn_layer = self.bottleneck_layer(encoded)

        cls_token_final = btn_layer[:, 0]  # just the CLS token
        cls = self.classifier(cls_token_final)

        for block_de in self.blocks_decoder:
            decoded = block_de(btn_layer)

        decoded = self.tanh(self.norm(decoded))

        return (decoded, cls)
