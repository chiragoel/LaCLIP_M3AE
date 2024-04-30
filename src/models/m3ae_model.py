import torch 
import torch.nn as nn
from torch.nn import functional as F

from ml_collections import ConfigDict
import transformers
import numpy as np

# @title PyTorch transformer definition
class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, depth, input_norm=True):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.input_norm = input_norm

        if self.input_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dense_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        x = inputs
        if self.input_norm:
            x = self.layer_norm(x)

        for i in range(self.depth):
            y = self.dense_layers[i](x)
            y = F.gelu(y)
            y = nn.LayerNorm(y)
            if i > 0:
                x = x + y
            else:
                x = y

        x = self.output_layer(x)
        return x


class DropPath(nn.Module):
    def __init__(self, dropout_prob=0.0):
        super(DropPath, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, input, deterministic=False, device='cpu'):
        if deterministic:
            return input

        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=torch.float32)
        random_tensor = random_tensor.floor().to(device)
        return input.div(keep_prob) * random_tensor


class TransformerMLP(nn.Module):
    def __init__(self, dim=256, out_dim=256, dropout=0.0, kernel_init=None):
        super(TransformerMLP, self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.kernel_init = kernel_init if kernel_init is not None else nn.init.xavier_uniform_

        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, out_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs, deterministic=False):
        x = self.fc1(inputs)
        x = F.gelu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.dropout_layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, use_bias=False, att_drop=0, proj_drop=0, kernel_init=None):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.att_drop = att_drop
        self.proj_drop = proj_drop
        self.scale = (dim // num_heads) ** -0.5
        self.kernel_init = kernel_init if kernel_init is not None else nn.init.xavier_uniform_

        self.qkv_linear = nn.Linear(dim, dim * 3, bias=use_bias)
        self.fc = nn.Linear(dim, dim)
        self.att_drop_layer = nn.Dropout(att_drop)
        self.proj_drop_layer = nn.Dropout(proj_drop)

    def forward(self, inputs, deterministic=False, padding_mask=None, device='cpu'):
        batch, n, channels = inputs.shape
        qkv = self.qkv_linear(inputs)
        qkv = qkv.view(batch, n, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            padding_mask = padding_mask.expand(attention.shape)
            # print('padding_mask',padding_mask)
            attention = torch.where(padding_mask > 0, torch.tensor(-1e7).to(device), attention)
            # print('attention', attention)

        attention = F.softmax(attention, dim=-1)
        attention = self.att_drop_layer(attention)

        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).reshape(batch, n, channels)
        x = self.fc(x)
        x = self.proj_drop_layer(x)
        return x


class Block(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, mlp_ratio=4, att_drop=0.0, drop=0.0, drop_path=0.0):
        super(Block, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.att_drop = att_drop
        self.drop = drop
        self.drop_path = drop_path

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.attention = Attention(emb_dim, num_heads, True, att_drop, drop)
        self.drop_path_layer1 = DropPath(drop_path)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.transformer_mlp = TransformerMLP(emb_dim, emb_dim, drop)
        self.drop_path_layer2 = DropPath(drop_path)

    def forward(self, inputs, deterministic=False, padding_mask=None, device='cpu'):
        x = self.layer_norm1(inputs)
        x = self.attention(x, deterministic, padding_mask, device)
        x = self.drop_path_layer1(x, device=device)
        inputs = inputs + x

        x = self.layer_norm2(inputs)
        x = self.transformer_mlp(x, deterministic)
        x = self.drop_path_layer2(x, device=device)
        return inputs + x


class Transformer(nn.Module):
    def __init__(self, emb_dim=1024, depth=24, att_drop=0, drop=0, drop_path=0, num_heads=16, mlp_ratio=4, device='cpu'):
        super(Transformer, self).__init__()
        self.emb_dim = emb_dim
        self.depth = depth
        self.att_drop = att_drop
        self.drop = drop
        self.drop_path = drop_path
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.device=device

        self.blocks = nn.ModuleList([
            Block(emb_dim, num_heads, mlp_ratio, att_drop, drop, drop_path)
            for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x, deterministic=False, padding_mask=None):
        for layer, block in enumerate(self.blocks):
            x = block(x, deterministic, padding_mask, device=self.device)

        x = self.layer_norm(x)
        return x

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return np.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, np.arange(length, dtype=np.float32)
        ),
        0
    )


def get_2d_sincos_pos_embed(embed_dim, length):
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return np.expand_dims(pos_embed, 0)

# @title Model size config
def get_transformer_by_config(model_type, config):
    if model_type == 'small':
        config.emb_dim = 384
        config.dec_emb_dim = 512
        config.depth = 12
        config.dec_depth = 8
        config.num_heads = 6
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'base':
        config.emb_dim = 768
        config.dec_emb_dim = 512
        config.depth = 12
        config.dec_depth = 8
        config.num_heads = 12
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'large':
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    else:
        raise ValueError('Unsupported model type!')

# @title PyTorch MaskedMultimodalAutoencoder
class MaskedMultimodalAutoencoder(nn.Module):
    def get_default_config(self, updates=None):
        config = ConfigDict()
        config.model_type = 'small'
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4

        config.output_head_depth = 0
        config.att_drop = 0.0
        config.drop = 0.0
        config.drop_path = 0.0

        config.use_type_embedding = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        if config.model_type is not None:
            get_transformer_by_config(config.model_type, config)

        return config

    def __init__(self, text_vocab_size, device, config_updates=None, depth=None, img_embed_dim=768):
        super(MaskedMultimodalAutoencoder, self).__init__()
        self.text_vocab_size = text_vocab_size
        self.config = self.get_default_config(config_updates)
        assert self.text_vocab_size > 0
        if depth is not None:
            self.config.depth = depth
        self.device = device
        self.text_embedding = nn.Embedding(self.text_vocab_size,
                                           self.config.emb_dim).to(device)
        self.text_embedding.weight.data.normal_(0.0, 1.0)
        self.image_embedding = nn.Linear(img_embed_dim, self.config.emb_dim).to(device)
        nn.init.xavier_uniform_(self.image_embedding.weight)

        if self.config.use_type_embedding:
            self.encoder_image_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
            ).to(device)
            self.encoder_text_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
            ).to(device)

        self.cls_token = nn.Parameter(
            torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
        ).to(device)

        self.encoder = Transformer(
            emb_dim=self.config.emb_dim,
            depth=self.config.depth,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            drop_path=self.config.drop_path,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            device=self.device
        )

    def get_type_embedding(self, name):
        if self.config.use_type_embedding:
            return {
                'encoder_image_type_embedding': self.encoder_image_type_embedding,
                'encoder_text_type_embedding': self.encoder_text_type_embedding,
            }[name]
        else:
            return 0.0

    def forward(self, image, text, text_padding_mask, deterministic=False):
        # print(image.shape, text.shape, text_padding_mask.shape)
        batch_size = image.shape[0]
        cls_token = self.cls_token.expand(batch_size, 1, self.config.emb_dim)
        input_tensors = [cls_token]
        padding_masks = [torch.zeros((batch_size, 1), dtype=torch.float32).to(self.device)]
        if image is not None:
            image_x = (
                self.image_embedding(image)
                + torch.from_numpy(get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1])).to(self.device)
                + self.get_type_embedding('encoder_image_type_embedding')
            )
            input_tensors.append(image_x)
            padding_masks.append(torch.zeros((batch_size, image.shape[1]), dtype=torch.float32).to(self.device))

        if text is not None:
            text_x = (
                self.text_embedding(text).squeeze(1)
                +  torch.from_numpy(get_1d_sincos_pos_embed(self.config.emb_dim, text.shape[1])).to(self.device)
                + self.get_type_embedding('encoder_text_type_embedding')
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask.to(self.device))

        # print(self.text_embedding(text).shape, torch.from_numpy(get_1d_sincos_pos_embed(self.config.emb_dim, text.shape[1])).shape)
        # print(image_x.shape, text_x.shape, cls_token.shape)
        x = torch.cat(input_tensors, dim=1)
        padding_mask = torch.cat(padding_masks, dim=1)
        x = self.encoder(x, deterministic, padding_mask)
        return x

        
class MMAEBase(nn.Module):
    def get_default_config(self, updates=None):
        config = ConfigDict()
        config.model_type = 'small'
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4

        config.output_head_depth = 0
        config.att_drop = 0.0
        config.drop = 0.0
        config.drop_path = 0.0

        config.use_type_embedding = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        if config.model_type is not None:
            get_transformer_by_config(config.model_type, config)

        return config

    def __init__(self, text_vocab_size, device, config_updates=None, img_embed_dim=3072,  layers=3, num_classes=2, model_type='base'):
        super(MMAEBase, self).__init__()
        self.text_vocab_size = text_vocab_size
        self.config = self.get_default_config(config_updates)
        assert self.text_vocab_size > 0
        self.device = device
        self.num_classes = num_classes
        self.text_embedding = nn.Embedding(self.text_vocab_size,
                                           self.config.emb_dim).to(device)
        self.text_embedding.weight.data.normal_(0.0, 1.0)
        self.image_embedding = nn.Linear(img_embed_dim, self.config.emb_dim).to(device)
        nn.init.xavier_uniform_(self.image_embedding.weight)

        if self.config.use_type_embedding:
            self.encoder_image_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
            ).to(device)
            self.encoder_text_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
            ).to(device)

        self.cls_token = nn.Parameter(
            torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
        ).to(device)

        self.encoder = self.load_embed_model(model_type, device, config_updates, layers)

        if model_type=='base':
            rep = 768
        elif model_type=='small':
            rep = 384
        elif model_type=='large':
            rep = 1024
        else:
            raise ValueError('Not valid model size in ViTclassifier')

        self.layernorm = nn.LayerNorm(rep)
        self.linear = nn.Linear(rep, self.num_classes)

    def load_embed_model(self, model_type, device, config_updates, layers):
        if model_type == 'base':
            embed_model_load_path = './torch_weights/m3ae_base.pth'
        elif args.model_type == 'small':
            embed_model_load_path = './torch_weights/m3ae_small.pth'
        else:
            embed_model_load_path = './torch_weights/m3ae_large.pth'

        map_location = device + ':0' if device == 'cuda' else device
        encoder_model = MaskedMultimodalAutoencoder(30522, device, config_updates)
        encoder_model.to(device)
        check_point_embed_model = torch.load(embed_model_load_path, map_location=map_location)
        encoder_model.load_state_dict(check_point_embed_model, strict=False)
        if layers is None:
            return encoder_model
        transfer_weights = nn.ModuleList([encoder_model.encoder.blocks[i] for i in range(layers)])
        transfer_weights.to(device)
        

        embed_model = Transformer(
            emb_dim=self.config.emb_dim,
            depth=layers,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            drop_path=self.config.drop_path,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            device=self.device
        )
        embed_model.to(device)
        embed_model.blocks.load_state_dict(transfer_weights.state_dict())
        return embed_model

    def get_type_embedding(self, name):
        if self.config.use_type_embedding:
            return {
                'encoder_image_type_embedding': self.encoder_image_type_embedding,
                'encoder_text_type_embedding': self.encoder_text_type_embedding,
            }[name]
        else:
            return 0.0

    def forward(self, image, text, text_padding_mask, deterministic=False, features=False, global_pool=True):
        batch_size = image.shape[0]
        # print('in forward',image.shape, text.shape, text_padding_mask.shape)
        cls_token = self.cls_token.expand(batch_size, 1, self.config.emb_dim)
        input_tensors = [cls_token]
        padding_masks = [torch.zeros((batch_size, 1), dtype=torch.float32).to(self.device)]
        if image is not None:
            image_x = (
                self.image_embedding(image)
                + torch.from_numpy(get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1])).to(self.device)
                + self.get_type_embedding('encoder_image_type_embedding')
            )
            input_tensors.append(image_x)
            padding_masks.append(torch.zeros((batch_size, image.shape[1]), dtype=torch.float32).to(self.device))

        if text is not None:
            text_x = (
                self.text_embedding(text).squeeze(1)
                +  torch.from_numpy(get_1d_sincos_pos_embed(self.config.emb_dim, text.shape[1])).to(self.device)
                + self.get_type_embedding('encoder_text_type_embedding')
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask.to(self.device))

        x = torch.cat(input_tensors, dim=1)
        padding_mask = torch.cat(padding_masks, dim=1)
        x = self.encoder(x, deterministic, padding_mask)
        # print('After Encoder', x.shape)
        if global_pool:
            x = x[:, 1:, :].mean(axis=1)  # global pool without cls token
        else:
            x = x[:, 0]

        # print('After Pool', x.shape)
        x = F.relu(self.layernorm(x))
        x = self.linear(x)
        log_probs = F.log_softmax(x, dim=1)
        # print(x,log_probs)

        if features:
            return log_probs, x
        else:
            return x