import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
import torchvision
import kornia
import cv2

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
    

class Dilated_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1))

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out
    

class Dilated_NoResblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_NoResblock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1))

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)

        return out
    

class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)
    """
    x: b,c,h,w
    y: b,c,2h,2w

    q:b,h,c,w
    k:b,2h,c,2w
    v:b,2h,c,2w

    qk:b
    """

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer
    

class CHGM(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(CHGM, self).__init__()

        self.conv_head1 = Depth_conv(in_ch=in_channels, out_ch=out_channels)
        self.conv_head2 = Depth_conv(in_ch=1, out_ch=out_channels)

        self.dilated_block = Dilated_Resblock(out_channels, out_channels)

        self.cross_attention1 = cross_attention(out_channels, num_heads=8)

        self.conv_tail = Depth_conv(out_channels, in_channels)
    
    def forward(self, fea, light):
        x1 = self.conv_head1(fea)
        x2 = self.conv_head2(light)

        y1 = self.cross_attention1(x2, x1)

        fix_fea = self.dilated_block(y1)
        out = self.conv_tail(fix_fea)

        return out

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    

class FeedForward(nn.Module):
    def __init__(self, in_channel, out_channel, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(in_channel * mult, in_channel * mult, 3, 1, 1,
                      bias=False, groups=in_channel * mult),
            GELU(),
            nn.Conv2d(in_channel * mult, out_channel, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)
    

class MSA(nn.Module):
    def __init__(
            self,
            in_channel,
            dim_head,
            heads=2,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(in_channel, dim_head * heads, bias=False)
        self.to_k = nn.Linear(in_channel, dim_head * heads, bias=False)
        self.to_v = nn.Linear(in_channel, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, in_channel, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False, groups=in_channel),
            GELU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False, groups=in_channel),
        )

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp))
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out
    

class AttBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.channel_level = 16

        self.ln1 = nn.LayerNorm(in_channel)
        self.MSA = MSA(in_channel=in_channel, dim_head=16, heads=in_channel//self.channel_level)
        self.ln2 = nn.LayerNorm(in_channel)
        self.ff = FeedForward(in_channel=in_channel, out_channel=out_channel)
    
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        x = self.MSA(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        out = x.permute(0, 3, 1, 2)
        return out
    

class ConvAttBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_feat):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_feat = n_feat

        self.conv1 = nn.Conv2d(self.in_channel, self.n_feat, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(self.n_feat + self.n_feat, self.out_channel, 1, 1, bias=True)
        self.AttBlock = AttBlock(self.n_feat, self.n_feat)

        self.conv_block = Dilated_Resblock(in_channels=n_feat, out_channels=n_feat)
    
    def forward(self, x):
        fea = self.conv1(x)
        conv_x = self.conv_block(fea)
        att_x = self.AttBlock(fea)
        out = self.conv2(torch.cat([conv_x, att_x], 1))

        return out


class BaseNet(nn.Module):

    def __init__(self, in_channel=3, out_channel=1, n_feat=32):
        super().__init__()
        self.head = Depth_conv(in_channel, n_feat) 
        self.conv_block1 = Dilated_NoResblock(n_feat, n_feat)
        self.cablock = CHGM(in_channels=n_feat)
        self.conv_block2 = Dilated_NoResblock(n_feat, n_feat)
        self.tail = Depth_conv(n_feat, out_channel) 
    
    def forward(self, x, light):
        x1 = self.head(x)
        x2 = self.conv_block1(x1)
        x3 = self.cablock(x2, light)
        x4 = self.conv_block2(x3)
        map = self.tail(x4)

        return map
    

class Mutil(nn.Module):

    def __init__(self, in_channel=8, out_channel=3, n_feat=16):
        super().__init__()
        self.head = Depth_conv(in_channel, n_feat)
        self.convatt1 = ConvAttBlock(in_channel=n_feat, out_channel=n_feat, n_feat=n_feat)
        self.down1 = nn.Conv2d(n_feat, n_feat*2, 2, 2)
        self.convatt2 = ConvAttBlock(in_channel=n_feat*3, out_channel=n_feat*2, n_feat=n_feat*2)
        self.down2 = nn.Conv2d(n_feat*2, n_feat*4, 2, 2)
        self.convatt3 = ConvAttBlock(in_channel=n_feat*5, out_channel=n_feat*4, n_feat=n_feat*4)
        self.down3 = nn.Conv2d(n_feat*4, n_feat*8, 2, 2)
        self.body = ConvAttBlock(in_channel=n_feat*9, out_channel=n_feat*8, n_feat=n_feat*8)
        self.up1 = nn.ConvTranspose2d(n_feat*8, n_feat*4, 2, 2, 0, bias=False)
        self.convatt4 = ConvAttBlock(in_channel=n_feat*4, out_channel=n_feat*4, n_feat=n_feat*4)
        self.up2 = nn.ConvTranspose2d(n_feat*4, n_feat*2, 2, 2, 0, bias=False)
        self.convatt5 = ConvAttBlock(in_channel=n_feat*2, out_channel=n_feat*2, n_feat=n_feat*2)
        self.up3 = nn.ConvTranspose2d(n_feat*2, n_feat, 2, 2, 0, bias=False)
        self.convatt6 = ConvAttBlock(in_channel=n_feat, out_channel=n_feat, n_feat=n_feat)
        self.tail = Depth_conv(n_feat, out_channel)

        self.conv1 = nn.Conv2d(n_feat*8, n_feat*4, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(n_feat*4, n_feat*2, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(n_feat*2, n_feat*1, 1, 1, bias=False)

        self.conv_block1 = Dilated_NoResblock(n_feat*4, n_feat*4)
        self.conv_block2 = Dilated_NoResblock(n_feat*2, n_feat*2)
        self.conv_block3 = Dilated_NoResblock(n_feat, n_feat)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, snr, color, map):
        x = torch.cat([x, snr, color, map], 1)
        x_br1 = F.interpolate(x, scale_factor=0.5, mode='bicubic', align_corners=True)
        x_br2 = F.interpolate(x_br1, scale_factor=0.5, mode='bicubic', align_corners=True)
        x_br3 = F.interpolate(x_br2, scale_factor=0.5, mode='bicubic', align_corners=True)

        x1 = self.head(x)
        x_b1 = self.head(x_br1)
        x_b2 = self.head(x_br2)
        x_b3 = self.head(x_br3)

        x2 = self.convatt1(x1)
        y1 = self.conv_block3(x2)
        x3 = self.down1(x2)
        x4 = self.convatt2(torch.cat([x3, x_b1], 1))
        y2 = self.conv_block2(x4)
        x5 = self.down2(x4)
        x6 = self.convatt3(torch.cat([x5, x_b2], 1))
        y3 = self.conv_block1(x6)
        x7 = self.down3(x6)
        x8 = self.body(torch.cat([x7, x_b3], 1))
        x9 = self.up1(x8)
        fusion1 = self.conv1(torch.cat([x9, y3], 1))
        x10 = self.convatt4(fusion1)
        x11 = self.up2(x10)
        fusion2 = self.conv2(torch.cat([x11, y2], 1))
        x12 = self.convatt5(fusion2)
        x13 = self.up3(x12)
        fusion3 = self.conv3(torch.cat([x13, y1], 1))
        x14 = self.convatt6(fusion3)
        out = self.tail(x14)

        return out
    

class RMCMNet(nn.Module):
    def __init__(self, istrain=1):
        super().__init__()
        self.Light = BaseNet()
        self.Denoise = Mutil()
        self.istrain = istrain

    def get_snr_mask(self, dark):
        light = kornia.filters.gaussian_blur2d(dark, (5, 5), (1.5, 1.5))
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0.0, max=1.0)
        return mask.float()
    
    def get_color_mask(self, dark):
        mean_rgb = torch.mean(dark, [2, 3], keepdim=True) 
        color_mask = dark / (mean_rgb + 0.0001)

        mask = torch.clamp(color_mask, min=0.0, max=1.0)
        return mask

    def forward(self, l_in):
        _, _, l_v = torch.split(kornia.color.rgb_to_hsv(l_in), 1, dim=1) 
        map = self.Light(l_in, l_v)
        restore_l = l_in * map
        snr_mask = self.get_snr_mask(restore_l)
        color_mask = self.get_color_mask(restore_l)
        noise = self.Denoise(restore_l, snr_mask, color_mask, map)
        result_l = restore_l - noise * map
        return restore_l, result_l





