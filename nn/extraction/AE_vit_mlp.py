# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103

import numpy as np
import torch
import torch.nn as nn

import config.config_train as config
from nn.tools.rmsnorm_torch import RMSNorm

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


class Attention(nn.Module):
    """Attention mechanism.
    the original code is from link below but its for linux
        https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

    this code is for windows
    """

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
        # self.norm = nn.LayerNorm(hidden_features, eps=1e-6)
        self.norm = RMSNorm(hidden_features)

    def forward(self, x):
        x = self.fc1(x)  # (n_samples, n_patches , hidden_features)
        x = self.act(x)  # (n_samples, n_patches , hidden_features)
        x = self.drop(x)  # (n_samples, n_patches , hidden_features)
        # this added as extra to improve
        # x = self.norm(x)
        x = self.fc2(x)  # (n_samples, n_patches , out_features)
        x = self.drop(x)  # (n_samples, n_patches , out_features)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, embed_size):
        super().__init__()
        self.sequence_length = sequence_length
        self.output_dim = 1
        # The inputs are of shape: `(batch_size, frames, num_features)
        self.positions = torch.tensor(
            [
                int(val)
                for val in torch.range(start=0, end=self.sequence_length - 1, step=1)
            ]
        ).to(device)
        self.position_embeddings = nn.Embedding(self.sequence_length, self.output_dim)

    def forward(self, inputs):
        return inputs + self.position_embeddings(self.positions)


class Block_Encoder(nn.Module):
    """Transformer block."""

    def __init__(self, dim, n_heads, mlp_ratio=4, qkv_bias=True, p=0.0, attn_p=0.0):
        super(Block_Encoder, self).__init__()
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # self.norm = RMSNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # one extra norm is added here to imporve
        # x = x + self.norm1(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class Block_Decoder(nn.Module):
    """Transformer block."""

    def __init__(self, dim, n_heads, mlp_ratio=4, qkv_bias=True, p=0.0, attn_p=0.0):
        super(Block_Decoder, self).__init__()
        ratio = 1

        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.attn_enc_to_dec = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )
        self.linear = nn.Linear(dim, input_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = x + self.attn_enc_to_dec(x)
        x = x + self.mlp(self.norm3(x))
        x = self.linear(x)
        return x


class AutoEncoderViTMLP(nn.Module):
    """Simplified implementation of the Vision transformer."""

    def __init__(self):
        super(AutoEncoderViTMLP, self).__init__()
        self.input_size = input_size
        self.sequences = sequence_length
        self.device = device
        ratio = 4  # 4 is best for gan
        dp = 0.2  # 0.2

        self.pos_embed = nn.Parameter(
            torch.randn(1, sequence_length, self.input_size)
        ).to(device)
        # self.pos_embed = PositionalEmbedding(self.sequences, self.input_size)
        self.pos_drop = nn.Dropout(0.1)
        self.tanh = nn.Tanh()
        self.glu = nn.GELU()

        self.kernel_div = 10
        # self.avgpool1 = nn.AvgPool2d((input_size,self.kernel_div))
        self.avgpool1 = nn.BatchNorm1d(input_size)
        self.avgpool2 = nn.AvgPool2d((sequence_length, 1))

        self.blocks_encoder = nn.ModuleList(
            [
                Block_Encoder(
                    dim=self.input_size,
                    n_heads=2,
                    mlp_ratio=2,
                    qkv_bias=True,
                    p=0.1,
                    attn_p=0.1,
                )
                for _ in range(2)
            ]
        )

        # self.blocks_decoder = nn.ModuleList(
        #                                     [
        #                                         Block_Decoder(
        #                                                 dim=self.input_size,
        #                                                 n_heads=n_heads,
        #                                                 mlp_ratio=mlp_ratio,
        #                                                 qkv_bias=qkv_bias,
        #                                                 p=p,
        #                                                 attn_p=attn_p,
        #                                             )
        #                                         for _ in range(depth)
        #                                     ])

        self.decoder = MLP(
            in_features=self.input_size,
            hidden_features=self.input_size * ratio,
            out_features=self.input_size,
            p=dp,
        )

        # self.decoder = nn.Sequential(
        #                                 nn.Conv1d(1, 16),
        #                                 activation_function,
        #                                 dropout,
        #                                 nn.Conv1d(16, 8),
        #                                 activation_function,
        #                                 dropout,
        #                                 nn.AvgPool1d(3)
        #                             )

        self.norm = nn.LayerNorm(self.input_size, eps=1e-6)
        self.norm_seq2 = nn.LayerNorm(self.input_size, eps=1e-6)
        self.norm_seq1 = nn.LayerNorm(self.sequences, eps=1e-6)
        #
        # self.norm = RMSNorm(self.input_size)
        # self.norm_seq2 = RMSNorm(self.input_size)
        # self.norm_seq1 = RMSNorm(self.sequences)

        self.mlp = MLP(
            in_features=self.input_size,
            hidden_features=self.input_size * ratio,
            out_features=self.input_size,
            p=dp,  # set it 0.2
        )

        self.classifier_mlp = nn.Sequential(nn.Linear(self.input_size, num_class))

        self.classifier_avg = nn.Sequential(
            nn.Linear(
                self.input_size + int(self.sequences / self.kernel_div), num_class
            )
        )

    def convert_to_norm_spect(self, x):
        list_img = []
        for i, img in enumerate(x):
            img = img.reshape(1, self.sequences, self.input_size)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            dft = torch.fft.fft2(img)
            dft_shift = torch.fft.fftshift(dft)
            phase_spectrum = torch.angle(dft_shift)
            magnitude_spectrum = 20 * torch.log(torch.abs(dft_shift))
            magnitude_spectrum_min = torch.amin(magnitude_spectrum, dim=(1, 2))
            magnitude_spectrum_max = torch.amax(magnitude_spectrum, dim=(1, 2))
            magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum_min) / (
                magnitude_spectrum_max - magnitude_spectrum_min
            )
            img_f = magnitude_spectrum.reshape(self.sequences, self.input_size)
            list_img.append(img_f)

        xx = torch.stack(list_img)
        return xx

    def avgpool_cls(self, norm_layer):
        final_head = norm_layer
        final_head1 = final_head.reshape(
            final_head.shape[0], final_head.shape[2], final_head.shape[1]
        )
        final_head1 = self.avgpool1(final_head1)
        final_head1 = final_head1.reshape(
            final_head.shape[0], int(self.sequences / self.kernel_div)
        )
        final_head2 = self.avgpool2(final_head)
        final_head2 = final_head2.reshape(final_head.shape[0], self.input_size)
        final_head = torch.cat((final_head1, final_head2), dim=1)
        final_head = final_head.reshape(
            final_head.shape[0], self.input_size + int(self.sequences / self.kernel_div)
        )
        return final_head

    def avgpool_with_fft_cls(self, norm_layer, x):
        final_head = norm_layer
        final_head1 = final_head.reshape(
            final_head.shape[0], final_head.shape[2], final_head.shape[1]
        )
        final_head1 = self.avgpool1(final_head1)
        final_head1 = final_head1.reshape(
            final_head.shape[0], int(self.sequences / self.kernel_div)
        )
        final_head2 = self.avgpool2(final_head)
        final_head2 = final_head2.reshape(final_head.shape[0], self.input_size)
        final_head = torch.cat((final_head1, final_head2), dim=1)
        final_head = final_head.reshape(
            final_head.shape[0], self.input_size + int(self.sequences / self.kernel_div)
        )

        xf0 = self.convert_to_norm_spect(x)
        for block in self.blocks_encoder:
            xf = block(xf0)

        xf1 = self.norm_seq1(xf.reshape(xf.shape[0], xf.shape[2], xf.shape[1]).float())
        xf2 = self.norm_seq2(xf.float())
        xf1 = self.avgpool1(xf1)
        xf1 = xf1.reshape(xf1.shape[0], xf1.shape[2])
        xf2 = self.avgpool2(xf2)
        xf2 = xf2.reshape(xf2.shape[0], xf2.shape[2])
        xf = torch.cat((xf1, xf2), dim=1)
        final_head = final_head + xf
        return final_head

    def mlp_cls(self, norm_layer):
        final_head = norm_layer[:, 0]  # just the last layer
        mlp_head = self.mlp(final_head)
        return mlp_head + final_head

    def forward(self, x):
        # x => (batch, sequences, features)
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        # x = self.pos_embed(x)
        for block in self.blocks_encoder:
            encoded = block(x)

        norm_layer = self.norm(encoded)

        final_head = self.mlp_cls(norm_layer)
        cls = self.classifier_mlp(final_head)

        # final_head = self.avgpool_cls(norm_layer)
        # #final_head = self.avgpool_with_fft_cls(norm_layer, x)
        # cls = self.classifier_avg(final_head)

        # for block in self.blocks_decoder:
        #     decoded = block(norm_layer)

        decoded = self.tanh(self.decoder(norm_layer))

        return (decoded, cls)
