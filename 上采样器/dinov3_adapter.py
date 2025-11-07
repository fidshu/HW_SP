"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from ..core import register
# from .vit_tiny import VisionTransformer
from .vit_base import VisionTransformer
#from .vit_large import VisionTransformer
from .dinov3 import DinoVisionTransformer
from ..deim import get_upsampler


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4


@register()
class DINOv3STAs(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_sta=True,
        conv_inplane=16,
        hidden_dim=None,
    ):
        super(DINOv3STAs, self).__init__()
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                checkpoint = torch.load(weights_path)
                if('dinov3_vitl16' in name):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint['teacher']
                
                # 去除 'backbone.' 前缀
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        k = k[9:]  # 去掉 'backbone.'
                    
                    # # 特殊处理 patch_embed.proj.weight
                    # if k == 'patch_embed.proj.weight' and v.shape[1] == 1:
                    #     # 从 [1024, 1, 16, 16] repeat 到 [1024, 3, 16, 16]
                    #     v = v.repeat(1, 3, 1, 1)
                    
                    new_state_dict[k] = v
                self.dinov3.load_state_dict(new_state_dict, strict=False)
            else:
                print('Training DINOv3 from scratch...')
        else:
            self.dinov3 = VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                checkpoint = torch.load(weights_path)
                state_dict = checkpoint['teacher']
                
                # 去除 'backbone.' 前缀
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        new_state_dict[k[9:]] = v  # 去掉 'backbone.' (9个字符)
                    else:
                        new_state_dict[k] = v
                
                self.dinov3._model.load_state_dict(new_state_dict, strict=True)
            else:
                print('Training ViT-Tiny from scratch...')
            # self.dinov3 =  VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            # if weights_path is not None and os.path.exists(weights_path):
            #     print(f'Loading ckpt from {weights_path}...')
            #     self.dinov3._model.load_state_dict(torch.load(weights_path)['teacher'])
            # else:
            #     print('Training ViT-Tiny from scratch...')

        self.jbu_in_conv = nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1, bias=False)
        self.jbu_upsampler = get_upsampler("jbu_one", dim=512)
        ckpt = torch.load("/yinghepool/BKNian/pth/xclip_jbu_one_million_aid.ckpt")['state_dict']
        weights_dict = {k[10:]: v for k, v in ckpt.items()}
        self.jbu_upsampler.load_state_dict(weights_dict, strict=True)
        print("[JBU] weights loaded")
        self.jbu_out_conv = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.SyncBatchNorm(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def forward(self, x):
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:    # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]
        
        sem_feats = []
        num_scales = len(all_layers) - 2
        jbu_upsampler = self.jbu_upsampler
        # 上采样
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            # print(f"\n>>> Layer {i} before upsample: {sem_feat.shape}")  # 打印上采样前的特征尺寸
            # 替换上采样
            scale = 2**(num_scales-i)
            resize_H, resize_W = int(H_c * scale), int(W_c * scale)
            if scale > 1:
                # print("===============")
                # print(sem_feat.size())
                # print("===============")
                # exit()
                
                f512 = self.jbu_in_conv(sem_feat)
                # print("after 192-->512",f512.shape)
                f_up = jbu_upsampler(f512, x.to(dtype=sem_feat.dtype))
                # print("after upsample",f_up.shape)
                sem_feat = self.jbu_out_conv(f_up)
                # print("after 512-->192",sem_feat.shape)
                # exit()
                # print(f"after upsample: {sem_feat.shape}  (×{scale} scale via JBUOne)")
            else:
                sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
                # print(f"after upsample: {sem_feat.shape}  (×{scale} scale bilinear)")
            sem_feats.append(sem_feat)
        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4