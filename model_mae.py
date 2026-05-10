# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
import torch.nn.functional as F

from util.pos_embed import get_2d_sincos_pos_embed
import ammd

# class MLPact(nn.Module):
#     def __init__(self, in_dim, batch_dim, out_dim):
#         super(MLPact, self).__init__()
#         self.linear = nn.Linear(in_dim, out_dim, bias=False)
#         self.bn = nn.BatchNorm1d(batch_dim, affine=True)

#     def forward(self, x):
#         x = self.linear(x)
#         out = F.relu(self.bn(x)) 
#         return out
        
# # Projector
# class projection_MLP_simsiam(nn.Module):
#     def __init__(self, in_dim, batch_dim=197, hidden_dim=256, out_dim=512):
#         super(projection_MLP_simsiam, self).__init__()
#         self.output_dim = out_dim
#         self.layer1 = MLPact(in_dim, batch_dim, 10)
#         self.layer2 = MLPact(batch_dim*10, hidden_dim, hidden_dim)
#         self.layer3 = nn.Linear(hidden_dim, out_dim, bias=False)
#         self.layer3_bn = nn.BatchNorm1d(out_dim, affine=False)

#     def forward(self, x):
#         x = self.layer1(x)
#         B, N, L = x.shape
#         x = x.reshape(B, N*L)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer3_bn(x)
#         return x 

# # Predictor 
# class prediction_MLP_simsiam(nn.Module):
#     def __init__(self, in_dim=512, hidden_dim=512, out_dim=512): 
#         super(prediction_MLP_simsiam, self).__init__()
#         self.layer1 = MLPact(in_dim, hidden_dim, hidden_dim)
#         self.layer2 = nn.Linear(hidden_dim, out_dim)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         return x 

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mask_ratio=0.75):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # # Projector
        # self.projector = projection_MLP_simsiam(embed_dim, batch_dim=int(num_patches * (1-mask_ratio))+1, hidden_dim=256, out_dim=512)

        # # Predictor
        # self.predictor = prediction_MLP_simsiam(in_dim=self.projector.output_dim, out_dim=self.projector.output_dim)
            
        self.mask_ratio = mask_ratio

        self.patch_size = patch_size

        self.mask_noise = None

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        if not isinstance(noise, torch.Tensor):
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, noise=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if not isinstance(noise, torch.Tensor):
            x, mask, ids_restore = self.random_masking(x)
        else:
            x, mask, ids_restore = self.random_masking(x, noise)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, global_latent=None, prev_latent=None):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        align_latent = None
        if isinstance(self.mask_noise, torch.Tensor):
            align_latent, _, _ = self.forward_encoder(imgs, self.mask_noise)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        mmd = torch.zeros((), device=loss.device, dtype=loss.dtype)

        # if isinstance(global_latent, torch.Tensor):
        #     # local branch to train: keep grad
        #     z_local = align_latent if isinstance(align_latent, torch.Tensor) else latent
        #     # global branch: NEVER backprop
        #     z_global = global_latent.detach()

        #     # reshape to [B, D] 
        #     # x = z_local.reshape(-1, z_local.size(-1))
        #     # y = z_global.reshape(-1, z_global.size(-1))

        #     sigma = ammd.mean_pairwise_distance(z_local.detach().float())  
        #     mmd = ammd.mmd_loss(z_local, z_global, sigma) 

        if isinstance(global_latent, torch.Tensor):
            latent_2 = global_latent.clone().requires_grad_(True)
            if isinstance(align_latent, torch.Tensor):
                latent_1 = align_latent.clone().requires_grad_(True)
            else:
                latent_1 = latent.clone().requires_grad_(True)
                # print("latent_1.requires_grad:", latent_1.requires_grad)
                # print("latent_2.requires_grad:", latent_2.requires_grad)
                # z1 = self.projector(latent_1)
                # p1 = self.predictor(z1)
                # z2 = self.projector(latent_2)
                # p2 = self.predictor(z2)

                # L = - (F.cosine_similarity(p1, z2.detach(), dim=-1).mean() +
                #     F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2

                # latent_1 = F.normalize(latent_1, p=2, dim=-1)  # L2 normalization
                # latent_2 = F.normalize(latent_2, p=2, dim=-1)  # L2 normalization

                # L = 1 - F.cosine_similarity(latent_1, latent_2, dim=-1).mean()
            # sigma = ammd.median_pairwise_distance(latent_1)
            sigma = ammd.mean_pairwise_distance(latent_1)
            mmd = ammd.mmd_loss(latent_1, latent_2, sigma)

            if isinstance(prev_latent, torch.Tensor):
                latent_3 = prev_latent.clone().requires_grad_(True)
                neg_mmd = ammd.mmd_loss(latent_1, latent_3, sigma)
                mmd = torch.nn.functional.softplus(mmd - neg_mmd)
                # eps = 1e-12
                # mmd =  torch.log((mmd + eps) / (neg_mmd + eps))

            # loss = loss + self.gamma * L

        if isinstance(align_latent, torch.Tensor):
            out_latent = align_latent.clone().detach()
        else:
            out_latent = latent.clone().detach()
        # out_latent = None
        # print("out_latent.requires_grad:", out_latent.requires_grad)

        return loss, mmd, pred, mask, out_latent
    
    def set_align_mask_noise(self, mask_noise):
        self.mask_noise = mask_noise

    # def forward(self, imgs, mask_ratio=0.75):
    #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    #     pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
    #     loss = self.forward_loss(imgs, pred, mask)
    #     return loss, pred, mask



def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

def mae_vit_tiny_patch8_dec512d1b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=768, depth=7, num_heads=12,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_tiny_patch8 = mae_vit_tiny_patch8_dec512d1b  # decoder: 512 dim, 1 block

def mae_vit_base_patch16_dec512d8b_nodp(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_patch16_nodp = mae_vit_base_patch16_dec512d8b_nodp  # decoder: 512 dim, 8 blocks

def mae_vit_base_patch16_decd1b_nodp(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, num_heads=12,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_p16_nodp_d1b = mae_vit_base_patch16_decd1b_nodp  # decoder: 512 dim, 1 block

def mae_vit_base_patch16_enc_dec_nodp(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_p16_nodp = mae_vit_base_patch16_enc_dec_nodp  

def mae_vit_tiny_patch16_decd1b_nodp(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=320, num_heads=16,
        decoder_embed_dim=128, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_tiny_p16_nodp = mae_vit_tiny_patch16_decd1b_nodp  # decoder: 512 dim, 1 block


