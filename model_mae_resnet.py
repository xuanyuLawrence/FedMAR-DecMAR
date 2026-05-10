import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import random
from typing import Callable, List, Optional, Type
from torch import Tensor
import torch.nn.functional as F

"""ResNet Implementation From https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18"""
"""ResNet Autoencoder Implementation From https://github.com/eleannavali/resnet-18-autoencoder"""

# Masking Function
def mask_image(images, device, mask_ratio=0.75, patch_size=16):
    """
    Mask images by patches.
    Args:
        images: A tensor of shape (batch_size, 3, H, W).
        device: The device to run the computation on.
        patch_size: The size of the patch (patch_size x patch_size).
        mask_ratio: The fraction of patches to mask.
    Returns:
        masked_images: Masked images of the same shape as input (batch_size, 3, H, W).
        masks: Binary masks of shape (batch_size, 1, H // patch_size, W // patch_size), indicating which patches are masked (1=masked, 0=unmasked).
    """
    batch_size, _, h, w = images.size()

    # Ensure the image dimensions are divisible by patch_size
    assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by patch_size."

    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    total_patches = num_patches_h * num_patches_w
    num_masked_patches = int(mask_ratio * total_patches)

    # Create random masks for the patches
    masks = torch.ones((batch_size, num_patches_h, num_patches_w), dtype=torch.float32).to(device)
    for i in range(batch_size):
        mask_indices = torch.randperm(total_patches)[:num_masked_patches]
        masks[i].view(-1)[mask_indices] = 0  # Flatten and apply masking

    # Expand masks to match image dimensions (batch_size, 1, H, W)
    masks = masks.unsqueeze(1).repeat(1, 1, patch_size, patch_size)
    masks = masks.view(batch_size, 1, h, w)

    # Apply masks to images
    masked_images = images * masks  # Element-wise masking

    return masked_images, masks



######### ResNet Encoder class #########

# def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
#     """3x3 convolution with padding"""
#     return nn.Conv2d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation,
#     )

# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlockEnc(nn.Module):
#     """The basic block architecture of resnet-18 network.
#     """
#     expansion: int = 1

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class ResNetEncoder(nn.Module):
#     """The encoder model.
#     """
#     def __init__(
#         self,
#         block: Type[BasicBlockEnc],
#         layers: List[int],
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation: Optional[List[bool]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
        
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 f"or a 3-element tuple, got {replace_stride_with_dilation}"
#             )
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, BasicBlockEnc) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

#     def _make_layer(
#         self,
#         block: Type[BasicBlockEnc],
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         dilate: bool = False,
#     ) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                 )
#             )

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x: Tensor) -> Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         return x

#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)

        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut_conv(x)
        if self.use_shortcut:
            shortcut = self.shortcut_bn(shortcut)
        out += shortcut
        return F.relu(out) 

class ResNetEncoder(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

######### ResNet Decoder class #########

def conv3x3Transposed(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, output_padding: int = 0) -> nn.Conv2d:
    """3x3 convolution with padding
    """
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding = output_padding, # output_padding is neccessary to invert conv2d with stride > 1
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1Transposed(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0) -> nn.Conv2d:
    """1x1 convolution
    """
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, output_padding = output_padding)


class BasicBlockDec(nn.Module):
    """The basic block architecture of resnet-18 network.
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        output_padding: int = 0,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3Transposed(planes, inplanes, stride, output_padding=output_padding)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3Transposed(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out



class ResNetDecoder(nn.Module):
    """The decoder model.
    """
    def __init__(
        self,
        block: Type[BasicBlockDec],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64 # change from 2048 to 64. It should be the shape of the output image chanel.
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.de_conv1 = nn.ConvTranspose2d(self.inplanes, 3, kernel_size=7, stride=2, padding=3, bias=False, output_padding=1)
        self.bn1 = norm_layer(3)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear') # NOTE: invert max pooling

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1 ,output_padding = 0, last_block_dim=64)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockDec) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlockDec],
        planes: int,
        blocks: int,
        stride: int = 2,
        output_padding: int = 1, # NOTE: output_padding will correct the dimensions of inverting conv2d with stride > 1.
        # More info:https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        last_block_dim: int = 0,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation

        layers = []

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        if last_block_dim == 0:
            last_block_dim = self.inplanes//2

        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                conv1x1Transposed(planes * block.expansion, last_block_dim, stride, output_padding),
                norm_layer(last_block_dim),
            )

        layers.append( block(
                last_block_dim, planes, stride, output_padding, upsample, self.groups, self.base_width, previous_dilation, norm_layer
            ))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.unpool(x)
        x = self.de_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

######### ResNet MaskedAutoencoder class #########
class ResNetMaskedAutoencoder(nn.Module):

    def __init__(self, num_layers=18, input_dim=224):
        """Initialize the autoencoder.
        Args:
            num_layers (int): the number of layers to be created. Choices [18, 34].
        """
        super().__init__()
        self.learnable_embedding = torch.nn.Parameter(torch.zeros(1, 3, input_dim, input_dim))

        if num_layers==18:
            # resnet 18 encoder
            self.encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2]) 
            # resnet 18 decoder
            self.decoder = ResNetDecoder(BasicBlockDec, [2, 2, 2, 2]) 
        elif num_layers==34:
            # resnet 34 encoder
            self.encoder = ResNetEncoder(BasicBlock, [3, 4, 6, 3]) 
            # resnet 34 decoder
            self.decoder = ResNetDecoder(BasicBlockDec, [3, 4, 6, 3]) 
        else:
            raise NotImplementedError("Only resnet 18 & 34 autoencoder have been implemented for images size >= 64x64.")

    def process_with_learnable_embedding(self, images, masks):
        """
        Process the images with the learnable embedding for masked parts.
        Args:
            images: A tensor of shape (batch_size, 3, H, W).
            masks: Binary masks of shape (batch_size, 1, H, W), indicating masked regions.
        Returns:
            combined_images: Images combined with learnable embeddings for masked regions.
        """
        batch_size = images.size(0)

        # Expand the learnable embedding to match the batch size
        learnable_embedding_expanded = self.learnable_embedding.expand(batch_size, -1, -1, -1)

        # Extract the masked regions by multiplying the mask with the learnable embedding
        masked_parts = (1 - masks) * self.learnable_embedding

        # Combine the masked parts with the unmasked parts of the original image
        unmasked_parts = images * masks
        combined_images = masked_parts + unmasked_parts

        return combined_images

    def forward(self, x, masks):
        x = self.process_with_learnable_embedding(x, masks)
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        # Apply the mask to the reconstructed output
        return reconstructed

# Loss Function
def reconstruction_loss(pred, target, mask):
    """Compute reconstruction loss for masked regions."""
    pred = pred * (1 - mask) + target * mask
    return ((pred - target) ** 2).mean()

class newResNet(nn.Module):

    def __init__(
        self,
        num_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if num_layers==18:
            self.encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2]) 
        elif num_layers==34:
            self.encoder = ResNetEncoder(BasicBlock, [3, 4, 6, 3]) 
        else:
            raise NotImplementedError("Only resnet 18 & 34 autoencoder have been implemented for images size >= 64x64.")
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.encoder(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



