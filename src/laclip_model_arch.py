# from functools import partial
#
# import torch
# import torch.nn as nn
#
# import numpy as np
# from collections import OrderedDict
#
# from timm.models.layers import Mlp, DropPath
# import timm.models.vision_transformer
#
#
# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""
#
#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)
#
#
# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)
#
#
# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#
#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.num_heads = n_head
#         self.attn_mask = attn_mask
#
#     # def attention(self, x: torch.Tensor):
#     #     self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#     #     return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
#
#     def forward(self, x: torch.Tensor, key_pad_mask: torch.Tensor):
#         if key_pad_mask is not None:
#             key_pad_mask = key_pad_mask.to(dtype=torch.bool, device=x.device)
#         else:
#             key_pad_mask = None
#         if self.attn_mask is not None:
#             self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)  # (256,77,77)
#         else:
#             self.attn_mask = None
#         ln_1_output = self.ln_1(x)
#         attn_output = self.attn(ln_1_output, ln_1_output, ln_1_output, need_weights=False, key_padding_mask=key_pad_mask, attn_mask=self.attn_mask)[0]
#         x = x + attn_output
#         x = x + self.mlp(self.ln_2(x))
#
#         return x
#
#
# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#     # def __init__(self, width: int, layers: int, heads: int):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
#
#     def forward(self, x: torch.Tensor, key_pad_mask: torch.Tensor = None):
#
#         for block in self.resblocks:
#             x = block(x, key_pad_mask)
#         return x
#
#
# class CLIP(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  vision_width: int,
#                  vision_model: nn.Module,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int,
#                  **kwargs,
#                  ):
#         super().__init__()
#
#         self.context_length = context_length
#         self.vision_width = vision_width
#
#         self.visual = vision_model
#
#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             # attn_mask=self.build_attention_mask(),
#             attn_mask=None
#         )
#
#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.ln_final = LayerNorm(transformer_width)
#
#         self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#
#         self.initialize_parameters()
#
#     def initialize_parameters(self):
#         nn.init.normal_(self.token_embedding.weight, std=0.02)
#         nn.init.normal_(self.positional_embedding, std=0.01)
#
#         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         attn_std = self.transformer.width ** -0.5
#         fc_std = (2 * self.transformer.width) ** -0.5
#         for block in self.transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
#
#         nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
#         nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
#
#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask
#
#     def encode_image(self, image):
#         x = self.visual(image)
#         # print("image after assing through ViT model :", x.shape) #(32, 50, 768) (1+49 cls_token/patch_Token)
#         last_hidden_state = x
#         proj_last_hidden_state = x @ self.image_projection # x.(768, 512) -> (32,50,512)
#         pooled_output = last_hidden_state[:, 0, :] #taking cls_token from ViT
#
#         return last_hidden_state, pooled_output, proj_last_hidden_state
#
#     def encode_text(self, text, key_pad_mask):
#         x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
#         x = x + self.positional_embedding
#         x = x.permute(1, 0, 2)  # NLD -> LND # (77, 32, 512)
#         x = self.transformer(x, key_pad_mask) # (77, 32, 512)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x)
#
#         last_hidden_state = x
#         proj_last_hidden_state = x @ self.text_projection
#
#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         # x = x[torch.arange( x.shape[0]), text.argmax(dim=-1)] @ self.text_projection #proj feature rep
#
#         pooled_output = x[torch.arange( x.shape[0]), text.argmax(dim=-1)]
#
#         return last_hidden_state, pooled_output, proj_last_hidden_state
#
#     def forward(self, image, text, attention_mask):
#         attention_mask = attention_mask.to('cpu')
#         attention_mask= torch.from_numpy(
#             np.where((attention_mask == 0) | (attention_mask == 1), attention_mask ^ 1,
#                      attention_mask)).to('mps')
#         image_last_hidden_state, image_pooled_output, image_proj_last_hidden_state = self.encode_image(image)
#         # text_last_hidden_state, text_pooled_output, text_proj_last_hidden_state = self.encode_text(text, None) #** no keypadmask
#         text_last_hidden_state, text_pooled_output, text_proj_last_hidden_state = self.encode_text(text, attention_mask) ## with keypadmask
#
#         return {'image_last_hidden_state': image_last_hidden_state,
#                 'text_last_hidden_state': text_last_hidden_state,
#                 'image_pooled_output': image_pooled_output,
#                 'text_pooled_output': text_pooled_output,
#                 'image_proj_last_hidden_state': image_proj_last_hidden_state,
#                 'text_proj_last_hidden_state': text_proj_last_hidden_state,
#                 'logit_scale': self.logit_scale.exp()}
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         # todo: add q and k norm for training stability
#         self.q_norm = nn.LayerNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
#         self.k_norm = nn.LayerNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
#
#         # todo: apply LN on query and key
#         q = self.q_norm(q)
#         k = self.k_norm(k)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class Block(nn.Module):
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
#                               attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#
#
# class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
#     """ Vision Transformer with support for global average pooling
#     """
#     def __init__(self, qk_norm=False, **kwargs):
#         super(VisionTransformer, self).__init__(**kwargs)
#         if qk_norm:
#             del self.blocks
#             embed_dim = kwargs['embed_dim']
#             num_heads = kwargs['num_heads']
#             mlp_ratio = kwargs['mlp_ratio']
#             qkv_bias = kwargs['qkv_bias']
#             depth = kwargs['depth']
#             drop_rate = 0.
#             attn_drop_rate = 0.
#             drop_path_rate = 0.
#             dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#             norm_layer = partial(nn.LayerNorm, eps=1e-6)
#             act_layer = nn.GELU
#             self.blocks = nn.Sequential(*[
#                 Block(
#                     dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
#                     act_layer=act_layer)
#                 for i in range(depth)])
#
#
# def vit_base_patch16_224(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
# def vit_base_patch32_224(**kwargs):
#     #changing global pool to '' from 'token'
#     model = VisionTransformer(
#         patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, global_pool='', qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
# def CLIP_VITB16(**kwargs):
#     vision_model = vit_base_patch16_224(qk_norm=True, num_classes=0)
#     model = CLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
#         transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)
#
#     return model
#
# def CLIP_VITB32(**kwargs):
#     vision_model = vit_base_patch32_224(qk_norm=True, num_classes=0)
#     model = CLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
#         transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)
#
#     return model
# #
# # from functools import partial
# #
# # import torch
# # import torch.nn as nn
# #
# # import numpy as np
# # from collections import OrderedDict
# #
# # from timm.models.layers import Mlp, DropPath
# # import timm.models.vision_transformer
# #
# #
# # class LayerNorm(nn.LayerNorm):
# #     """Subclass torch's LayerNorm to handle fp16."""
# #
# #     def forward(self, x: torch.Tensor):
# #         orig_type = x.dtype
# #         ret = super().forward(x.type(torch.float32))
# #         return ret.type(orig_type)
# #
# #
# # class QuickGELU(nn.Module):
# #     def forward(self, x: torch.Tensor):
# #         return x * torch.sigmoid(1.702 * x)
# #
# #
# # class ResidualAttentionBlock(nn.Module):
# #     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
# #         super().__init__()
# #
# #         self.attn = nn.MultiheadAttention(d_model, n_head)
# #         self.ln_1 = LayerNorm(d_model)
# #         self.mlp = nn.Sequential(OrderedDict([
# #             ("c_fc", nn.Linear(d_model, d_model * 4)),
# #             ("gelu", QuickGELU()),
# #             ("c_proj", nn.Linear(d_model * 4, d_model))
# #         ]))
# #         self.ln_2 = LayerNorm(d_model)
# #         self.attn_mask = attn_mask
# #
# #     def attention(self, x: torch.Tensor):
# #         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
# #         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
# #
# #     def forward(self, x: torch.Tensor):
# #         x = x + self.attention(self.ln_1(x))
# #         x = x + self.mlp(self.ln_2(x))
# #         return x
# #
# #
# # class Transformer(nn.Module):
# #     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
# #         super().__init__()
# #         self.width = width
# #         self.layers = layers
# #         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
# #
# #     def forward(self, x: torch.Tensor):
# #         return self.resblocks(x)
# #
# #
# # class CLIP(nn.Module):
# #     def __init__(self,
# #                  embed_dim: int,
# #                  # vision
# #                  vision_width: int,
# #                  vision_model: nn.Module,
# #                  # text
# #                  context_length: int,
# #                  vocab_size: int,
# #                  transformer_width: int,
# #                  transformer_heads: int,
# #                  transformer_layers: int,
# #                  **kwargs,
# #                  ):
# #         super().__init__()
# #
# #         self.context_length = context_length
# #         self.vision_width = vision_width
# #
# #         self.visual = vision_model
# #
# #         self.transformer = Transformer(
# #             width=transformer_width,
# #             layers=transformer_layers,
# #             heads=transformer_heads,
# #             attn_mask=self.build_attention_mask(),
# #         )
# #
# #         self.vocab_size = vocab_size
# #         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
# #         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
# #         self.ln_final = LayerNorm(transformer_width)
# #
# #         self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
# #         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
# #         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
# #
# #         self.initialize_parameters()
# #
# #     def initialize_parameters(self):
# #         nn.init.normal_(self.token_embedding.weight, std=0.02)
# #         nn.init.normal_(self.positional_embedding, std=0.01)
# #
# #         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
# #         attn_std = self.transformer.width ** -0.5
# #         fc_std = (2 * self.transformer.width) ** -0.5
# #         for block in self.transformer.resblocks:
# #             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
# #             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
# #             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
# #             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
# #
# #         nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
# #         nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
# #
# #     def build_attention_mask(self):
# #         # lazily create causal attention mask, with full attention between the vision tokens
# #         # pytorch uses additive attention mask; fill with -inf
# #         mask = torch.empty(self.context_length, self.context_length)
# #         mask.fill_(float("-inf"))
# #         mask.triu_(1)  # zero out the lower diagonal
# #         return mask
# #
# #     def encode_image(self, image):
# #         x = self.visual(image)
# #         # x = x @ self.image_projection #proj hidden state
# #
# #         last_hidden_state = x
# #         proj_last_hidden_state = x @ self.image_projection
# #         pooled_output = last_hidden_state[:, 0, :]
# #         # self.visual.fc_norm(pooled_output) can be applied, but
# #         # fc_norm is nn.Identity() for global_pool=''
# #
# #         return last_hidden_state, pooled_output, proj_last_hidden_state
# #
# #     def encode_text(self, text):
# #         x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
# #         x = x + self.positional_embedding
# #         x = x.permute(1, 0, 2)  # NLD -> LND
# #         x = self.transformer(x)
# #         x = x.permute(1, 0, 2)  # LND -> NLD
# #         x = self.ln_final(x)
# #
# #         last_hidden_state = x
# #         proj_last_hidden_state = x @ self.text_projection
# #
# #         # x.shape = [batch_size, n_ctx, transformer.width]
# #         # take features from the eot embedding (eot_token is the highest number in each sequence)
# #         # x = x[torch.arange( x.shape[0]), text.argmax(dim=-1)] @ self.text_projection #proj feature rep
# #
# #         pooled_output = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
# #
# #         return last_hidden_state, pooled_output, proj_last_hidden_state
# #
# #     def forward(self, image, text):
# #         image_last_hidden_state, image_pooled_output, image_proj_last_hidden_state = self.encode_image(image)
# #         text_last_hidden_state, text_pooled_output, text_proj_last_hidden_state = self.encode_text(text)
# #
# #         return {'image_last_hidden_state': image_last_hidden_state,
# #                 'text_last_hidden_state': text_last_hidden_state,
# #                 'image_pooled_output': image_pooled_output,
# #                 'text_pooled_output': text_pooled_output,
# #                 'image_proj_last_hidden_state': image_proj_last_hidden_state,
# #                 'text_proj_last_hidden_state': text_proj_last_hidden_state,
# #                 'logit_scale': self.logit_scale.exp()}
# #
# #
# # class Attention(nn.Module):
# #     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0.):
# #         super().__init__()
# #         self.num_heads = num_heads
# #         head_dim = dim // num_heads
# #         self.scale = head_dim ** -0.5
# #
# #         # todo: add q and k norm for training stability
# #         self.q_norm = nn.LayerNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
# #         self.k_norm = nn.LayerNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
# #
# #         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
# #         self.attn_drop = nn.Dropout(attn_drop)
# #         self.proj = nn.Linear(dim, dim)
# #         self.proj_drop = nn.Dropout(proj_drop)
# #
# #     def forward(self, x):
# #         B, N, C = x.shape
# #         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
# #         q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
# #
# #         # todo: apply LN on query and key
# #         q = self.q_norm(q)
# #         k = self.k_norm(k)
# #
# #         attn = (q @ k.transpose(-2, -1)) * self.scale
# #         attn = attn.softmax(dim=-1)
# #         attn = self.attn_drop(attn)
# #
# #         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
# #         x = self.proj(x)
# #         x = self.proj_drop(x)
# #         return x
# #
# #
# # class Block(nn.Module):
# #
# #     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, drop=0., attn_drop=0.,
# #                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
# #         super().__init__()
# #         self.norm1 = norm_layer(dim)
# #         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
# #                               attn_drop=attn_drop, proj_drop=drop)
# #         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
# #         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
# #         self.norm2 = norm_layer(dim)
# #         mlp_hidden_dim = int(dim * mlp_ratio)
# #         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
# #
# #     def forward(self, x):
# #         x = x + self.drop_path(self.attn(self.norm1(x)))
# #         x = x + self.drop_path(self.mlp(self.norm2(x)))
# #         return x
# #
# #
# # class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
# #     """ Vision Transformer with support for global average pooling
# #     """
# #
# #     def __init__(self, qk_norm=False, **kwargs):
# #         super(VisionTransformer, self).__init__(**kwargs)
# #         if qk_norm:
# #             del self.blocks
# #             embed_dim = kwargs['embed_dim']
# #             num_heads = kwargs['num_heads']
# #             mlp_ratio = kwargs['mlp_ratio']
# #             qkv_bias = kwargs['qkv_bias']
# #             depth = kwargs['depth']
# #             drop_rate = 0.
# #             attn_drop_rate = 0.
# #             drop_path_rate = 0.
# #             dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
# #             norm_layer = partial(nn.LayerNorm, eps=1e-6)
# #             act_layer = nn.GELU
# #             self.blocks = nn.Sequential(*[
# #                 Block(
# #                     dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
# #                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
# #                     act_layer=act_layer)
# #                 for i in range(depth)])
# #
# #
# # def vit_base_patch16_224(**kwargs):
# #     model = VisionTransformer(
# #         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
# #         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
# #     return model
# #
# #
# # def vit_base_patch32_224(**kwargs):
# #     # changing global pool to '' from 'token'
# #     model = VisionTransformer(
# #         patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, global_pool='', qkv_bias=True,
# #         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
# #     return model
# #
# #
# # def CLIP_VITB16(**kwargs):
# #     vision_model = vit_base_patch16_224(qk_norm=True, num_classes=0)
# #     model = CLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
# #                  transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)
# #
# #     return model
# #
# #
# # def CLIP_VITB32(**kwargs):
# #     vision_model = vit_base_patch32_224(qk_norm=True, num_classes=0)
# #     model = CLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
# #                  transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)
# #
# #     return model
# #

""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from open_clip.timm_model import TimmModel
from open_clip.utils import freeze_batch_norm_2d, to_2tuple


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, key_padding_mask=attn_mask)[0]
        # return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(
            self, image_size: int, patch_size: int, width: int, layers: int, heads: int, mlp_ratio: float,
            output_dim: int, act_layer: Callable = nn.GELU):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_post(x[:, 0, :])
        pooled_output = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            # x = x @ self.proj
            projected = x @ self.proj

        return x, pooled_output, projected


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12


class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.context_length = text_cfg.context_length

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELU if quick_gelu else nn.GELU

        if vision_cfg.timm_model_name:
            self.visual = TimmModel(
                vision_cfg.timm_model_name,
                pretrained=vision_cfg.timm_model_pretrained,
                pool=vision_cfg.timm_pool,
                proj=vision_cfg.timm_proj,
                embed_dim=embed_dim,
                image_size=vision_cfg.image_size
            )
            act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
        elif isinstance(vision_cfg.layers, (tuple, list)):
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
            self.visual = ModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
        else:
            vision_heads = vision_cfg.width // vision_cfg.head_width
            self.visual = VisualTransformer(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                heads=vision_heads,
                mlp_ratio=vision_cfg.mlp_ratio,
                output_dim=embed_dim,
                act_layer=act_layer,
            )

        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, text_cfg.width))
        self.ln_final = LayerNorm(text_cfg.width)

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if hasattr(self.visual, 'init_parameters'):
            self.visual.init_parameters()

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image):
        last_hidden_state, pooled_output, projection = self.visual(image)
        # # print("image after assing through ViT model :", x.shape) #(32, 50, 768) (1+49 cls_token/patch_Token)
        # last_hidden_state = x
        # proj_last_hidden_state = x@self.visual.proj.T # x.(768, 512) -> (32,50,512)
        # pooled_output = last_hidden_state[:, 0, :] #taking cls_token from ViT

        return last_hidden_state, pooled_output, projection
        # return self.visual(image)

    def encode_text(self, text, key_pad_mask):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=key_pad_mask)
        # x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        last_hidden_state = x
        proj_last_hidden_state = x @ self.text_projection
        pooled_output = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        return last_hidden_state, pooled_output, proj_last_hidden_state

        # return x

    def forward(self, image, text, attention_mask):
        attention_mask = attention_mask.to('cpu')
        attention_mask = torch.from_numpy(np.bool_(np.where((attention_mask == 0) | (attention_mask == 1), attention_mask ^ 1,attention_mask))).to('mps')

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        # image_features = self.encode_image(image)
        # image_features = F.normalize(image_features, dim=-1)
        #
        # text_features = self.encode_text(text, attention_mask)
        # text_features = F.normalize(text_features, dim=-1)

        image_last_hidden_state, image_pooled_output, image_proj_last_hidden_state = self.encode_image(image)
        # text_last_hidden_state, text_pooled_output, text_proj_last_hidden_state = self.encode_text(text, None) #** no keypadmask
        text_last_hidden_state, text_pooled_output, text_proj_last_hidden_state = self.encode_text(text, attention_mask)  ## with keypadmask

        return {'image_last_hidden_state': image_last_hidden_state,
                'text_last_hidden_state': text_last_hidden_state,
                'image_pooled_output': image_pooled_output,
                'text_pooled_output': text_pooled_output,
                'image_proj_last_hidden_state': image_proj_last_hidden_state,
                'text_proj_last_hidden_state': text_proj_last_hidden_state,
                'logit_scale': self.logit_scale.exp()}

        # return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model_from_openai_state_dict(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=True,  # OpenAI models were trained with QuickGELU
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed
