import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.non_local_simple_version import NONLocalBlock2D
from models.fasterkan import FasterKAN as KAN
import pywt
from einops import rearrange, reduce, repeat
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
"""
this code was based on MVCINN at https://github.com/XiaolingLuo/MVCINN

@inproceedings{luo2023MVCINN,
title={MVCINN: Multi-View Diabetic Retinopathy Detection Using a Deep Cross-Interaction Neural Network},
author={Luo, Xiaoling and Liu, Chengliang and Wong, Waikeung and Wen, Jie and Jin, Xiaopeng and Xu, Yong},
booktitle={Thirty-Seventh AAAI Conference on Artificial Intelligence},
year={2023}}

and our paper was accepted by IEEE ICME 2025
@inproceedings{xia2025ccsk,
title={Cross-Structure and Semantic Enhancement for Diabetic Retinopathy Grading},
author={Xia, Xue and Lin, Zipeng and Zhu, Jingying and Yan, Jiebin and Fang, Yuming},
booktitle={The 2025 IEEE International Conference on Multimedia and Expo {(ICME)}},
year={2025},
}
"""


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Linear layers for cross-attention
        self.q_ll = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_hl_lh = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_x_t = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self,x , ll=None, hl_lh=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_slef = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if ll is not None and hl_lh is not None:
            q_ll = self.q_ll(ll).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv_x_t = self.kv_x_t(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k_x_t, v_x_t = kv_x_t[0], kv_x_t[1]
            attn_ll = (q_ll @ k_x_t.transpose(-2, -1)) * self.scale
            attn_ll = attn_ll.softmax(dim=-1)
            attn_ll = self.attn_drop(attn_ll)
            x_cross_attn_ll = (attn_ll @ v_x_t).transpose(1, 2).reshape(B, N, C)

            # Cross-Attention with HL+LH as Q
            q_hl_lh = self.q_hl_lh(hl_lh).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            attn_hl_lh = (q_hl_lh @ k_x_t.transpose(-2, -1)) * self.scale
            attn_hl_lh = attn_hl_lh.softmax(dim=-1)
            attn_hl_lh = self.attn_drop(attn_hl_lh)
            x_cross_attn_hl_lh = (attn_hl_lh @ v_x_t).transpose(1, 2).reshape(B, N, C)

            # Combine self-attention and cross-attentions
            x = x_slef + x_cross_attn_ll + x_cross_attn_hl_lh
        else:
            x = x_slef
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU6, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None,type=0):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion
        self.type=type
        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)
        self.conv_med = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)
        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)
        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)
        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)
        return x


class FCUUp(nn.Module):
    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class CSKBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.normll = norm_layer(dim)
        self.normhl= norm_layer(dim)

        self.norm2 = norm_layer(dim)
        self.norm = norm_layer(dim)

        self.kan = KAN([dim, 20, dim])

    def forward(self, x, ll=None, hl_lh=None):
        b, t, d = x.shape
        if ll is not None and hl_lh is not None:
            x = x + self.drop_path(self.attn(self.norm1(x),self.normll(ll),self.normhl(hl_lh)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.kan(self.norm2(x).reshape(-1, x.shape[-1])).reshape(b, t, d))
        return x


class CCSKBlock(nn.Module):
    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1,T=0):
        super(CCSKBlock, self).__init__()
        expansion = 4
        self.view = 2

        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups,type=T)
        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True,
                                          groups=groups,type=T)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups,type=T)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.csk_block = CSKBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

        self.conv1x1 = nn.Conv2d(embed_dim, 3, kernel_size=1)

        self.linear_ll = nn.Linear(1, embed_dim)
        self.linear_hl_lh = nn.Linear(1, embed_dim)

        # A learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.conv_fuse_hl_lh = nn.Conv2d(1, 1, kernel_size=1)
        self.type = T

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)
        B, _, H, W = x2.shape
        if self.type == 1:
            feature_map_part = x_t[:, 1:, :]  # Shape: [B, H*W, dim]

            # Reshape the feature map part back to 4D
            feature_map = feature_map_part.view(B, H, W, self.embed_dim)  # Shape: [B, H, W, dim]

            # Permute dimensions to get [B, dim, H, W]
            feature_map = feature_map.permute(0, 3, 1, 2)  # Shape: cc

            # Apply a convolution to match the required number of output channels
            map = self.conv1x1(feature_map)

            # Apply SWT (Stationary Wavelet Transformation) on the Green-channel
            g_channel = map[:, 1:2, :, :]
            coeffs = pywt.swt2(g_channel.detach().cpu().numpy(), 'haar', level=1)
            ll, (cH, cV, cD) = coeffs[0]
            cH = torch.tensor(cH).to(x.device)
            cV = torch.tensor(cV).to(x.device)

            hl_lh = self.conv_fuse_hl_lh(cH + cV)

            ll = torch.tensor(ll).to(x.device)

            # Flatten to sequences
            ll_flat = ll.flatten(2).transpose(1, 2)
            hl_lh_flat = hl_lh.flatten(2).transpose(1, 2)
            # Dimension Expansion
            ll_flat = self.linear_ll(ll_flat)  # Shape: [B, 196, dim]
            hl_lh_flat = self.linear_hl_lh(hl_lh_flat)  # Shape: [B, 196, dim]

            # CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: [B, 1, dim]
            ll_cross = torch.cat((cls_tokens, ll_flat), dim=1)  # Shape: [B, 197, dim]
            hl_lh_cross = torch.cat((cls_tokens, hl_lh_flat), dim=1)  # Shape: [B, 197, dim]
            x_t = self.csk_block(x_t,ll_cross,hl_lh_cross)
        else:
            stack_ll = []
            stack_hllh = []
            x_t_ori = rearrange(x_t, '(b v) c e -> b v c e', v=2)
            for i in range(self.view):
                t = x_t_ori[:,i]
                feature_map_part = t[:, 1:, :]  # Shape: [B, H*W, dim]

                # Reshape the feature map part back to 4D
                feature_map = feature_map_part.view(B//self.view, 14, 14, self.embed_dim)  # Shape: [B, H, W, dim]

                # Permute dimensions to get [B, dim, H, W]
                feature_map = feature_map.permute(0, 3, 1, 2)  # Shape: cc

                # Apply a convolution to match the required number of output channels
                map = self.conv1x1(feature_map)

                # Apply SWT (Stationary Wavelet Transformation) on the Green-channel
                g_channel = map[:, 1:2, :, :]
                coeffs = pywt.swt2(g_channel.detach().cpu().numpy(), 'haar', level=1)
                ll, (cH, cV, cD) = coeffs[0]

                cH = torch.tensor(cH).to(x.device)
                cV = torch.tensor(cV).to(x.device)

                hl_lh = self.conv_fuse_hl_lh(cH + cV)
                ll = torch.tensor(ll).to(x.device)

                # Flatten to sequences
                ll_flat = ll.flatten(2).transpose(1, 2)
                hl_lh_flat = hl_lh.flatten(2).transpose(1, 2)

                # Dimension expansion
                ll_flat = self.linear_ll(ll_flat)  # Shape: [B, 196, dim]
                hl_lh_flat = self.linear_hl_lh(hl_lh_flat)  # Shape: [B, 196, dim]

                stack_ll.append(ll_flat)
                stack_hllh.append(hl_lh_flat)

            ll_flat = torch.cat(stack_ll, dim=0)
            hl_lh_flat = torch.cat(stack_hllh, dim=0)
            # CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: [B, 1, dim]
            ll_cross = torch.cat((cls_tokens, ll_flat), dim=1)  # Shape: [B, 197, dim]
            hl_lh_cross = torch.cat((cls_tokens, hl_lh_flat), dim=1)  # Shape: [B, 197, dim]
            x_t = self.csk_block(x_t, ll_cross, hl_lh_cross)
        x_st = self.squeeze_block(x2, x_t)

        x_tout = x_t + x_st
        # if no fusion for transformer branch, use it!
        # x_tout = x_t
        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        # if no fusion for convolution branch, mask it!
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_tout


class CCSKformer(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        print('new-model! ')
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0
        self.num_view = 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.csk_norm = nn.LayerNorm(embed_dim)
        self.csk_cls_head = nn.Linear(embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1st layer
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.csk_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.csk_1 = CSKBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4th layer
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_csk_' + str(i),
                            CCSKBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block,T=0
                            )
                            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8th layer
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_csk_' + str(i),
                            CCSKBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block,T=0
                            )
                            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)

        for i in range(0, self.num_view):
            s = 2
            in_channel = stage_2_channel
            res_conv = True
            last_fusion = True
            self.add_module('mv_models_' + str(i),
                            CCSKBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion,T=1 )
                            )

        self.fin_stage = fin_stage
        # self.nn.Parameter
        self.jointLayer = jointLayer(stage_3_channel)
        # print(jointLayer)
        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def joint(self, x):
        x = rearrange(x, '(b v) c h w -> b v c h w', v=2)
        b, v, c, h, w = x.shape
        arr = x
        arr = arr.view(b, c,  h, 2 * w)
        arr[:, :, :h, :w] = x[:, 0, :, :, :]
        arr[:, :, :h, w:2 * w] = x[:, 1, :, :, :]

        return arr

    def _add(self, x):
        x = rearrange(x, '(b v) c e -> b c (v e)', v=2)
        # x = rearrange(x, '(b v) c e -> b c v e', v=4)
        # x = torch.einsum('bcve->bce',x)

        return x

    def forward(self, x):
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.csk_patch_conv(x_base)  # [B*4, 576, 14, 14]
        x_t = x_t.flatten(2).transpose(1, 2)  # [B*4, 196, 576]
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.csk_1(x_t)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_csk_' + str(i))(x, x_t)

        x = rearrange(x, '(b v) c h w -> b v c h w', v=2)
        x_t = rearrange(x_t, '(b v) c e -> b v c e', v=2)

        mv_x = []
        mv_x_t = []
        for i in range(0, self.num_view):
            sv_x, sv_x_t = eval('self.mv_models_' + str(i))(x[:, i], x_t[:, i])
            mv_x.append(sv_x)
            mv_x_t.append(sv_x_t)

        x = torch.stack(mv_x, 1)
        x = rearrange(x, 'b v c h w -> (b v) c h w', v=2)
        x_t = torch.stack(mv_x_t, 1)
        x_t = rearrange(x_t, 'b v c e -> (b v) c e', v=2)


        x = self.jointLayer(x)
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)

        x_t = self.csk_norm(x_t)  # [12 197 576]
        x_t = self._add(x_t)
        cks_cls = self.csk_cls_head(x_t[:, 0])

        return [conv_cls, cks_cls], 1


class jointLayer(nn.Module):
    def __init__(self, in_channels=1536):
        super().__init__()
        self.NONLocalBlock2D = NONLocalBlock2D(in_channels=in_channels, sub_sample=True)

    def joint(self, x):
        x = rearrange(x, '(b v) c h w -> b v c h w', v=2)
        b, v, c, h, w = x.shape
        arr = x.clone()
        arr = arr.view(b, c, h, 2 * w)
        arr[:, :, :h, :w] = x[:, 0, :, :, :]
        arr[:, :, :h, w:2 * w] = x[:, 1, :, :, :]

        return arr

    def forward(self, x):
        # 1.concat
        x = self.joint(x)
        # 2.add
        # x = rearrange(x, '(b v) c h w -> b v c h w', v=4)
        # x = torch.einsum('bvchw->bchw',x)
        x = self.NONLocalBlock2D(x)
        return x