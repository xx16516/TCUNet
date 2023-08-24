import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_,to_2tuple
from torchinfo import summary
import math
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
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
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
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
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # print(k.shape,v.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
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
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
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
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W
class Conv2dReLU(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding=0,stride=1):# dilation=1,
        super(Conv2dReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False,)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6())
# class DecoderBlock(nn.Module):
#     def __init__( self,in_channels, skip_channels,out_channels):
#         super().__init__()
#         self.conv1 = Conv2dReLU(in_channels + skip_channels,out_channels,kernel_size=3, padding=1,)
#         self.conv2 = Conv2dReLU(out_channels,out_channels,kernel_size=3,padding=1,)
#     def forward(self, x, skip=None):
#         x = F.interpolate(x, scale_factor=2, mode="bilinear",align_corners=True)
#         if skip is not None:
#             x = torch.cat([skip,x], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x

class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #

        return x
class AlignedModulev2(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModulev2, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
    def forward(self, low,high):
        low_feature, h_feature = low,high
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :] , flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)
        fuse_feature = h_feature_warp + l_feature_warp

        return fuse_feature,l_feature_warp,h_feature_warp
    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_chan, skip_chan,out_chan,reduction_ratio=16):
        super(DecoderBlock, self).__init__()
        self.ca1 = Channel_Att(in_chan)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.fa1 = AlignedModulev2(out_chan,out_chan)
        self.ca2 = Channel_Att(out_chan)
        self.sa = nn.Sequential(
            nn.Conv2d(out_chan, out_chan//reduction_ratio, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(out_chan//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // reduction_ratio, out_chan // reduction_ratio, kernel_size=3,padding=4, dilation=4),
            nn.BatchNorm2d(out_chan // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan//16, out_chan, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, high,low):
        high = self.ca1(high)
        high = self.conv1(high)
        fuse ,low,high = self.fa1(low,high)
        # print(fuse.shape,high.shape,low.shape)
        f_c = self.ca2(fuse)
        f_s = self.sa(fuse)
        weight = torch.sigmoid(f_s+f_c)
        x = high*weight + (1-weight)*low
        return x
# class DecoderBlock(nn.Module):
#     def __init__( self,in_channels, skip_channels,out_channels, eps=1e-8):
#         super().__init__()
#         self.conv1 = Conv2dReLU( skip_channels,out_channels,kernel_size=3, padding=1,)
#         self.conv = Conv2dReLU(in_channels,skip_channels,kernel_size=1,padding=0)
#         self.fuse =  AlignedModule(skip_channels,out_channels)
#         # self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         # self.eps = eps
#     def forward(self, x, skip=None):
#         # print(x.shape,skip.shape)
#         # x = F.interpolate(x, scale_factor=2, mode="nearest")
#         # weights = nn.ReLU()(self.weights)
#         # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         if skip is not None:
#             x = self.fuse([skip,self.conv(x)])
#             # x = fuse_weights[0] * skip + fuse_weights[1] * self.conv(x)
#         x = self.conv1(x)
#         return x

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        expansion = 4
        med_planes = outplanes // expansion
        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=med_planes, padding=1, bias=False)
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

class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """
    def __init__(self, inplanes, outplanes,  act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()
        if inplanes == outplanes:
            self.k = 3
        else:
            self.k = 1
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=self.k, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x= self.conv_project(x)
        x = self.bn(x)
        x = self.act(x)
        # x_r = self.act(self.bn(self.conv_project(x)))
        return x

class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        if inplanes == outplanes:
            self.k = 3
        else:
            self.k = 1
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=self.k, stride=1, padding=0)
        # self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv_project(x)  # [N, C, H, W]
        # x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        return x
# from conformer_unet.conformer import Block as block
class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """
    def __init__(self, inplanes, outplanes, res_conv, stride,  embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, groups=1,sr_ratio=1):
        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)
        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)
        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim)
        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion)
        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,sr_ratio=sr_ratio)
        self.embed_dim = embed_dim
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)
        # print(x.shape,x2.shape)
        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2)
        x_t = self.trans_block(x_st + x_t,H,W)
        x_t_r = self.expand_block(x_t, H , W )
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t
class Conunetv1(nn.Module):

    def __init__(self, img_size=512, in_chans=8, num_classes=2, base_channel=32, channel_ratio=2, use_cnn = True,
                 embed_dim=32, depth=[3,4,6,3], num_heads=[1,2,4,8], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,sr_ratios=[8, 4, 2, 1]):
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
        super().__init__()
        # Transformer
        self.use_cnn =use_cnn
        self.n_channel = in_chans
        self.n_classes=self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.trans_dpr =  [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]   # stochastic depth decay rule

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv0 = nn.Conv2d(in_chans, base_channel, kernel_size=7, stride=2, padding=3,bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(base_channel)
        self.act1 = nn.ReLU(inplace=True)

        self.trans_patch_conv0 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=2, in_chans=in_chans,embed_dim=embed_dim)
        self.trans_0 = Block(dim=embed_dim, num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],sr_ratio=sr_ratios[0]*2
                             )
        self.norm0 = nn.LayerNorm(embed_dim)

        # 1 stage
        cnn_1_channel = int(base_channel * channel_ratio)
        trans_1_dim = int(embed_dim)*channel_ratio
        self.conv_1 = ConvBlock(inplanes=base_channel, outplanes=cnn_1_channel, res_conv=True, stride=2)
        self.trans_patch_conv1 = OverlapPatchEmbed(img_size=img_size//2, patch_size=3, stride=2, in_chans=embed_dim,embed_dim=trans_1_dim)
        self.trans_1 = Block(dim=trans_1_dim, num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],sr_ratio=sr_ratios[0]
                             )
        cur = 1
        self.block1 = nn.ModuleList([ConvTransBlock(inplanes=cnn_1_channel,outplanes=cnn_1_channel,embed_dim=trans_1_dim,res_conv=False, stride=1,
                                                    num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=self.trans_dpr[cur+i],sr_ratio=sr_ratios[0])
                                         for i in range(depth[0] - 1)])
        self.norm1 =nn.LayerNorm(trans_1_dim)

        # 2 stage
        cur += depth[0]
        cnn_2_channel = int(cnn_1_channel * channel_ratio)
        trans_2_dim = int(embed_dim)*2
        self.conv_2 = ConvBlock(inplanes=cnn_1_channel, outplanes=cnn_2_channel, res_conv=True, stride=2)
        self.trans_patch_conv2 = OverlapPatchEmbed(img_size=img_size//4, patch_size=3, stride=2, in_chans=trans_1_dim,embed_dim=trans_2_dim)
        self.trans_2 = Block(dim=trans_2_dim, num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[cur-1],sr_ratio=sr_ratios[1]
                             )
        self.block2 = nn.ModuleList([ConvTransBlock(inplanes=cnn_2_channel,outplanes=cnn_2_channel,embed_dim=trans_2_dim,res_conv=False, stride=1,
                                                    num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=self.trans_dpr[cur+i],sr_ratio=sr_ratios[1])
                                         for i in range(depth[1] - 1)])
        self.norm2 = nn.LayerNorm(trans_2_dim)

        # 3 stage
        cur += depth[1]
        cnn_3_channel = int(cnn_2_channel * channel_ratio)
        trans_3_dim = int(trans_2_dim)*2
        self.conv_3 = ConvBlock(inplanes=cnn_2_channel, outplanes=cnn_3_channel, res_conv=True, stride=2)
        self.trans_patch_conv3 = OverlapPatchEmbed(img_size=img_size//8, patch_size=3, stride=2, in_chans=trans_2_dim,embed_dim=trans_3_dim)
        self.trans_3 = Block(dim=trans_3_dim, num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[cur-1],sr_ratio=sr_ratios[2]
                             )
        self.block3 = nn.ModuleList([ConvTransBlock(inplanes=cnn_3_channel,outplanes=cnn_3_channel,embed_dim=trans_3_dim,res_conv=False, stride=1,
                                                    num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=self.trans_dpr[cur+i],sr_ratio=sr_ratios[2])
                                         for i in range(depth[2] - 1)])
        self.norm3 = nn.LayerNorm(trans_3_dim)

        # 4 stage
        cur += depth[2]
        cnn_4_channel = int(cnn_3_channel * channel_ratio)
        trans_4_dim = int(trans_3_dim)*2
        self.conv_4 = ConvBlock(inplanes=cnn_3_channel, outplanes=cnn_4_channel, res_conv=True, stride=2)
        self.trans_patch_conv4 = OverlapPatchEmbed(img_size=img_size//16, patch_size=3, stride=2, in_chans=trans_3_dim,embed_dim=trans_4_dim)
        self.trans_4 = Block(dim=trans_4_dim, num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[cur-1],sr_ratio=sr_ratios[3]
                             )
        self.block4 = nn.ModuleList([ConvTransBlock(inplanes=cnn_4_channel,outplanes=cnn_4_channel,embed_dim=trans_4_dim,res_conv=False, stride=1,
                                                    num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=self.trans_dpr[cur+i],sr_ratio=sr_ratios[3])
                                         for i in range(depth[3] - 1)])
        self.norm4 = nn.LayerNorm(trans_4_dim)

        # self.aspp = ASPP('conunet',8,nn.BatchNorm2d)

        #Decoder
        self.d4 = DecoderBlock(cnn_4_channel,cnn_3_channel,cnn_3_channel) if use_cnn else DecoderBlock(trans_4_dim,trans_3_dim,trans_3_dim)
        self.d3 = DecoderBlock(cnn_3_channel,cnn_2_channel,cnn_2_channel) if use_cnn else DecoderBlock(trans_3_dim,trans_2_dim,trans_2_dim)
        self.d2 = DecoderBlock(cnn_2_channel, cnn_1_channel, cnn_1_channel) if use_cnn else DecoderBlock(trans_2_dim, trans_1_dim, trans_1_dim)
        self.d1 = DecoderBlock(cnn_1_channel, base_channel, base_channel) if use_cnn else DecoderBlock(trans_1_dim, trans_1_dim, trans_1_dim)
        if use_cnn:
            self.segmentation_head = nn.Sequential(ConvBNReLU(base_channel, base_channel, 3),
                                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                   nn.Dropout2d(p=0.1, inplace=True),
                                                   nn.Conv2d(base_channel, num_classes, kernel_size=1))
        else:
            self.segmentation_head = nn.Sequential(ConvBNReLU(embed_dim, embed_dim, 3),
                                                # nn.ConvTranspose2d(embed_dim,embed_dim,2,2),
                                               nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                                               nn.Dropout2d(p=0.1, inplace=True),
                                               nn.Conv2d(embed_dim, num_classes, kernel_size=1))

    def forward(self, x):
        # Encoder
        B = x.shape[0]
        # h,w =x.shape[2]
        x_c = self.act1(self.bn1(self.conv0(x))) #B base_channel H W   CNN分支
        x_t, H, W = self.trans_patch_conv0(x)  # TRANS分支  B (H/2)*(W/2) embed_dim
        x_t = self.norm0(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e=x_c if self.use_cnn else x_t
        # 1 stage
        x_c = self.conv_1(x_c, return_x_2=False)  # B base_channel*2 H/2 W/2
        x_t,H,W= self.trans_patch_conv1(x_t) #TRANS分支  B (H/2)*(W/2) embed_dim
        x_t = self.trans_1(x_t,H,W) #TRANS分支 B (H/2)*(W/2) embed_dim
        for blk in self.block1:
            x_c, x_t = blk(x_c, x_t)
        x_t = self.norm1(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e0 = x_c if self.use_cnn else x_t

        # 2 stage
        x_c = self.conv_2(x_c, return_x_2=False)  # B base_channel*4 H/4 W/4
        x_t,H,W= self.trans_patch_conv2(x_t) #TRANS分支  B (H/2)*(W/2) embed_dim
        x_t = self.trans_2(x_t,H,W) #TRANS分支 B (H/2)*(W/2) embed_dim
        for blk in self.block2:
            x_c, x_t = blk(x_c, x_t)
        x_t = self.norm2(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e1 = x_c if self.use_cnn else x_t

        # 3 stage
        x_c = self.conv_3(x_c, return_x_2=False)  # B base_channel*4 H/4 W/4
        x_t,H,W= self.trans_patch_conv3(x_t) #TRANS分支  B (H/2)*(W/2) embed_dim
        x_t = self.trans_3(x_t,H,W) #TRANS分支 B (H/2)*(W/2) embed_dim
        for blk in self.block3:
            x_c, x_t = blk(x_c, x_t)
        x_t = self.norm3(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e2 = x_c if self.use_cnn else x_t

        # 4 stage
        x_c = self.conv_4(x_c, return_x_2=False)  # B base_channel*4 H/4 W/4
        x_t,H,W= self.trans_patch_conv4(x_t) #TRANS分支  B (H/2)*(W/2) embed_dim
        x_t = self.trans_4(x_t,H,W) #TRANS分支 B (H/2)*(W/2) embed_dim
        for blk in self.block4:
            x_c, x_t = blk(x_c, x_t)
        x_t = self.norm4(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e3 =  x_c if self.use_cnn else x_t
        #Decoder
        # e3 = self.aspp(e3)
        x = self.d4(e3,e2)
        del e3,e2
        x = self.d3(x,e1)
        del e1
        x = self.d2(x,e0)
        del e0
        x = self.d1(x,e)
        del e
        x = self.segmentation_head(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # print(x.shape)
        return x
class Conunet(nn.Module):

    def __init__(self, img_size=512, in_chans=8, num_classes=2, base_channel=32, channel_ratio=2, use_cnn = True,
                 embed_dim=32, depth=[3,4,6,3], num_heads=[1,2,4,8], mlp_ratio=4., qkv_bias=False, qk_scale=None,L = True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,sr_ratios=[8, 4, 2, 1]):
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
        super().__init__()
        # Transformer
        self.use_cnn =use_cnn
        self.n_channel = in_chans
        self.n_classes=self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.trans_dpr =  [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]   # stochastic depth decay rule

        # stage1: get the feature maps by conv block (copied form resnet.py)
        self.conv0 = nn.Conv2d(in_chans, base_channel, kernel_size=7, stride=2, padding=3,bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(base_channel)
        self.act1 = nn.ReLU(inplace=True)

        self.trans_patch_conv0 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=2, in_chans=in_chans,embed_dim=embed_dim)
        self.trans_0 = Block(dim=embed_dim, num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],sr_ratio=sr_ratios[0]
                             )
        self.norm0 = nn.LayerNorm(embed_dim)

        # 2 stage
        cnn_1_channel = int(base_channel * channel_ratio)
        trans_1_dim = int(embed_dim)*channel_ratio
        self.conv_1 = ConvBlock(inplanes=base_channel, outplanes=cnn_1_channel, res_conv=True, stride=2)
        self.trans_patch_conv1 = OverlapPatchEmbed(img_size=img_size//2, patch_size=3, stride=2, in_chans=embed_dim,embed_dim=trans_1_dim)
        self.trans_1 = Block(dim=trans_1_dim, num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],sr_ratio=sr_ratios[0]
                             )
        cur = 1
        self.block1 = nn.ModuleList([ConvTransBlock(inplanes=cnn_1_channel,outplanes=cnn_1_channel,embed_dim=trans_1_dim,res_conv=False, stride=1,
                                                    num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=self.trans_dpr[cur+i],sr_ratio=sr_ratios[0])
                                         for i in range(depth[0] - 1)])
        self.norm1 =nn.LayerNorm(trans_1_dim)

        # 3 stage
        cur += depth[0]
        cnn_2_channel = int(cnn_1_channel * channel_ratio)
        trans_2_dim = int(embed_dim)*2
        self.conv_2 = ConvBlock(inplanes=cnn_1_channel, outplanes=cnn_2_channel, res_conv=True, stride=2)
        self.trans_patch_conv2 = OverlapPatchEmbed(img_size=img_size//4, patch_size=3, stride=2, in_chans=trans_1_dim,embed_dim=trans_2_dim)
        self.trans_2 = Block(dim=trans_2_dim, num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[cur-1],sr_ratio=sr_ratios[1]
                             )
        self.block2 = nn.ModuleList([ConvTransBlock(inplanes=cnn_2_channel,outplanes=cnn_2_channel,embed_dim=trans_2_dim,res_conv=False, stride=1,
                                                    num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=self.trans_dpr[cur+i],sr_ratio=sr_ratios[1])
                                         for i in range(depth[1] - 1)])
        self.norm2 = nn.LayerNorm(trans_2_dim)

        # 4 stage
        cur += depth[1]
        cnn_3_channel = int(cnn_2_channel * channel_ratio)
        trans_3_dim = int(trans_2_dim)*2
        self.conv_3 = ConvBlock(inplanes=cnn_2_channel, outplanes=cnn_3_channel, res_conv=True, stride=2)
        self.trans_patch_conv3 = OverlapPatchEmbed(img_size=img_size//8, patch_size=3, stride=2, in_chans=trans_2_dim,embed_dim=trans_3_dim)
        self.trans_3 = Block(dim=trans_3_dim, num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[cur-1],sr_ratio=sr_ratios[2]
                             )
        self.block3 = nn.ModuleList([ConvTransBlock(inplanes=cnn_3_channel,outplanes=cnn_3_channel,embed_dim=trans_3_dim,res_conv=False, stride=1,
                                                    num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=self.trans_dpr[cur+i],sr_ratio=sr_ratios[2])
                                         for i in range(depth[2] - 1)])
        self.norm3 = nn.LayerNorm(trans_3_dim)

        # 5 stage
        cur += depth[2]
        cnn_4_channel = int(cnn_3_channel * channel_ratio)
        trans_4_dim = int(trans_3_dim)*2
        self.conv_4 = ConvBlock(inplanes=cnn_3_channel, outplanes=cnn_4_channel, res_conv=True, stride=2)
        self.trans_patch_conv4 = OverlapPatchEmbed(img_size=img_size//16, patch_size=3, stride=2, in_chans=trans_3_dim,embed_dim=trans_4_dim)
        self.trans_4 = Block(dim=trans_4_dim, num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[cur-1],sr_ratio=sr_ratios[3]
                             )
        self.block4 = nn.ModuleList([ConvTransBlock(inplanes=cnn_4_channel,outplanes=cnn_4_channel,embed_dim=trans_4_dim,res_conv=False, stride=1,
                                                    num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=self.trans_dpr[cur+i],sr_ratio=sr_ratios[3])
                                         for i in range(depth[3] - 1)])
        self.norm4 = nn.LayerNorm(trans_4_dim)


        #Decoder
        self.d4 = DecoderBlock(cnn_4_channel,cnn_3_channel,cnn_3_channel) if use_cnn else DecoderBlock(trans_4_dim,trans_3_dim,trans_3_dim)
        self.d3 = DecoderBlock(cnn_3_channel,cnn_2_channel,cnn_2_channel) if use_cnn else DecoderBlock(trans_3_dim,trans_2_dim,trans_2_dim)
        self.d2 = DecoderBlock(cnn_2_channel, cnn_1_channel, cnn_1_channel) if use_cnn else DecoderBlock(trans_2_dim, trans_1_dim, trans_1_dim)
        self.d1 = DecoderBlock(cnn_1_channel, base_channel, base_channel) if use_cnn else DecoderBlock(trans_1_dim, embed_dim, embed_dim)
        if use_cnn:
            self.segmentation_head = nn.Sequential(ConvBNReLU(base_channel, base_channel, 3),
                                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                   nn.Dropout2d(p=0.1, inplace=True),
                                                   nn.Conv2d(base_channel, num_classes, kernel_size=1))
        else:
            self.segmentation_head = nn.Sequential(ConvBNReLU(embed_dim, embed_dim, 3),
                                               nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                                               nn.Dropout2d(p=0.1, inplace=True),
                                               nn.Conv2d(embed_dim, num_classes, kernel_size=1))

    def forward(self, x):
        # Encoder
        B = x.shape[0]
        # 1 stage
        x_c = self.act1(self.bn1(self.conv0(x))) #B base_channel H W   CNN分支
        x_t, H, W = self.trans_patch_conv0(x)  # TRANS分支  B (H/2)*(W/2) embed_dim
        x_t = self.trans_0(x_t,H,W)
        x_t = self.norm0(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e=x_c if self.use_cnn else x_t
        # 2 stage
        x_c = self.conv_1(x_c, return_x_2=False)  # B base_channel*2 H/4 W/4
        x_t,H,W= self.trans_patch_conv1(x_t) #TRANS分支  B (H/2)*(W/2) embed_dim
        x_t = self.trans_1(x_t,H,W)
        for blk in self.block1:
            x_c, x_t = blk(x_c, x_t)
        x_t = self.norm1(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e0 = x_c if self.use_cnn else x_t

        # 3 stage
        x_c = self.conv_2(x_c, return_x_2=False)  # B base_channel*4 H/4 W/4
        x_t,H,W= self.trans_patch_conv2(x_t) #TRANS分支  B (H/2)*(W/2) embed_dim
        x_t = self.trans_2(x_t,H,W) #TRANS分支 B (H/2)*(W/2) embed_dim
        for blk in self.block2:
            x_c, x_t = blk(x_c, x_t)
        x_t = self.norm2(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e1 = x_c if self.use_cnn else x_t

        # 4 stage
        x_c = self.conv_3(x_c, return_x_2=False)
        x_t,H,W= self.trans_patch_conv3(x_t)
        x_t = self.trans_3(x_t,H,W)
        for blk in self.block3:
            x_c, x_t = blk(x_c, x_t)
        x_t = self.norm3(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e2 = x_c if self.use_cnn else x_t

        # 5 stage
        x_c = self.conv_4(x_c, return_x_2=False)
        x_t,H,W= self.trans_patch_conv4(x_t)
        x_t = self.trans_4(x_t,H,W)

        for blk in self.block4:
            x_c, x_t = blk(x_c, x_t)
        x_t = self.norm4(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e3 =  x_c if self.use_cnn else x_t

        #Decoder

        x = self.d4(e3,e2)
        del e3,e2
        x = self.d3(x,e1)
        del e1
        x = self.d2(x,e0)
        del e0
        x = self.d1(x,e)
        del e
        x = self.segmentation_head(x)
        return x


if __name__ == '__main__':

    a = 256
    model = Conunet(img_size=a,depth=[3,4,6,3],use_cnn=True,base_channel=16,embed_dim=16)
    model1 = FCUDown(8,16)
    model2 = ConvBlock(8,16,res_conv=True)
    data = torch.randn(16, 8, a, a)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (8, a, a), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    summary(model, input_size=data.shape)