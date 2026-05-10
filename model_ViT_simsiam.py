import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class MLPact(nn.Module):
    def __init__(self, in_dim, batch_dim, out_dim):
        super(MLPact, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(batch_dim, affine=True)

    def forward(self, x):
        x = self.linear(x)
        out = F.relu(self.bn(x)) 
        return out
        
# Projector
class projection_MLP_simsiam(nn.Module):
    def __init__(self, in_dim, batch_dim=197, hidden_dim=256, out_dim=512):
        super(projection_MLP_simsiam, self).__init__()
        self.output_dim = out_dim
        self.layer1 = MLPact(in_dim, batch_dim, 10)
        self.layer2 = MLPact(batch_dim*10, hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer3_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = self.layer1(x)
        B, N, L = x.shape
        x = x.reshape(B, N*L)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3_bn(x)
        return x 

# Predictor 
class prediction_MLP_simsiam(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512): 
        super(prediction_MLP_simsiam, self).__init__()
        self.layer1 = MLPact(in_dim, hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

# SimSiam with ViT using timm
class ViT_Simsiam(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(ViT_Simsiam, self).__init__()

        # Create backbone
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Projector
        self.projector = projection_MLP_simsiam(embed_dim, batch_dim=num_patches+1, hidden_dim=256, out_dim=512)

        # Predictor
        self.predictor = prediction_MLP_simsiam(in_dim=self.projector.output_dim, out_dim=self.projector.output_dim)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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

    def forward_backbone(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        return x
    
    def forward(self, x1, x2, x3=None, deg_labels=None):
        # Pass inputs through the backbone and projector
        b1 = self.forward_backbone(x1)
        z1 = self.projector(b1)
        z2 = self.projector(self.forward_backbone(x2))
        
        # Pass projected features through the predictor
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # Compute the loss
        L = - (F.cosine_similarity(p1, z2.detach(), dim=-1).mean() +
               F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2

        return L


def mae_vit_base_patch16_enc_dec_nodp(**kwargs):
    model = ViT_Simsiam(
        patch_size=16, embed_dim=768, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_p16_nodp = mae_vit_base_patch16_enc_dec_nodp 

def mae_vit_tiny_patch16_decd1b_nodp(**kwargs):
    model = ViT_Simsiam(
        patch_size=14, embed_dim=320, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_tiny_p16_nodp = mae_vit_tiny_patch16_decd1b_nodp  # decoder: 512 dim, 1 block
