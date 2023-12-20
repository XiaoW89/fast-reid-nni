import math
import torch
import logging
import torch.nn as nn
import pdb

from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from fastreid.utils import comm

import nni 
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, Cell

from fastreid.layers import (
    IBN,
    SELayer,
    Non_local,
    get_norm,
)

logger = logging.getLogger(__name__)


def _make_divisible(v, divisor, min_value=None, limit=False, ratio=0.9): # 0.9
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if limit:
        if new_v < ratio * v:
            new_v += divisor

    return new_v


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ConvWithIBN(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvWithIBN, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = IBN(c2, nn.BatchNorm2d)
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ELANBlock(nn.Module):
    def __init__(self, inp, hip, oup, layer_num, concat_id, act=True, with_ibn=False):
        super(ELANBlock, self).__init__()
        assert layer_num >= 2, "ELANBlock: layer num should be bigger or equal than 2."
        assert concat_id[-1] + layer_num == 0, "{} {}: concat layer id is out of range.".format(layer_num, concat_id)

        self.concat_id = [layer_num + i for i in concat_id]
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        self.conv1 = Conv(inp, hip, 1, 1, act=self.act) if not with_ibn else ConvWithIBN(inp, hip, 1, 1, act=self.act)
        layers = [Conv(inp, hip, 1, 1, act=self.act)]
        for i in range(2, layer_num):
            layers.append(Conv(hip, hip, 3, 1, act=self.act))
        self.features = nn.Sequential(*layers)

        self.concat = Concat(dimension=1)
        self.conv = Conv(hip * layer_num, oup, 1, 1, act=self.act)

    def forward(self, x):
        outs = []

        y1 = self.conv1(x)
        outs.append(y1)

        y2 = x
        for id, mod in enumerate(self.features):
            y2 = mod(y2)
            if (id + 1) in self.concat_id:
                 outs.append(y2)

        out = self.concat(outs)
        out = self.conv(out)

        return out


class ELANNas(nn.Module):
    def __init__(self, input_channel=32, pretrained=None, with_ibn=False, feat_dim = 512):
        super(ELANNas, self).__init__()
        mp = MP
        block = ELANBlock
        elan_block_setting = [
            # hidden_channel, out_channel, layer_num, concat_id
            [  32,   64, 4, [-1, -2, -3, -4]],
            [  64, 128, 4, [-1, -2, -3, -4]],
            [128, 256, 4, [-1, -2, -3, -4]],
            [256, 256, 4, [-1, -2, -3, -4]],
            [256, feat_dim, 4, [-1, -2, -3, -4]]
        ]

        # building first layer
        input_channel = _make_divisible(input_channel,  8)

        inp = 32
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=autopad(3),
                               bias=False)
        self.bn1 = get_norm("BN", 32)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = LayerChoice([nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                    nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                    Conv(inp, inp, 3, 2, p=autopad(3), act=True),
                                    ], label='pool_1')
        self.cell1 = Cell(op_candidates=[nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=autopad(3)), 
                           nn.MaxPool2d(kernel_size=3, stride=1, padding=autopad(3)),
                           nn.AvgPool2d(kernel_size=3, stride=1, padding=autopad(3)),
#                           nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=3*(3-1)/2, dilation=3),
#                           nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=5*(3-1)/2, dilation=5),
                           nn.Identity(),
                           ], 
                          num_nodes=3,
                          num_ops_per_node=2,
                          num_predecessors=1,
                          merge_op='loose_end',
                          label='cell_1'
                          )
        inp = len(self.cell1.output_node_indices) * input_channel
        self.layer1, inp = self._make_layer(elan_block_setting, 0, inp, num_block=1)
        self.layer2, inp = self._make_layer(elan_block_setting, 1, inp, num_block=1)
        self.layer3, inp = self._make_layer(elan_block_setting, 2, inp, num_block=1)
        self.layer4, inp = self._make_layer(elan_block_setting, 3, inp, num_block=2)
        self.layer5, inp = self._make_layer(elan_block_setting, 4, inp, num_block=2)


        #self.fpn = 

        # building elan blocks
#        for i in range(len(elan_block_setting)):
#            hc, oc, ln, ci = elan_block_setting[i]
#            hip = _make_divisible(hc, 8)
#            oup = _make_divisible(oc, 8)
#            if i > 0:
#                layers.append(mp(k=2))
#            layers.append(block(inp, hip, oup, ln, ci, with_ibn=with_ibn))
##            if i > 0:
##                layers.append(Non_local(oup, "BN"))
#            inp = oup

        # make it nn.Sequential
        self.init_weights(pretrained=pretrained)
#        self.features = nn.Sequential(*layers)

#    def _make_fpn(self, layer_configs):
#        for config in layer_configs:
#            inp, oup = config
            

    def _make_layer(self, setting, i, inp, with_ibn=False, num_block=1):
        layers = []
        hc, oc, ln, ci = setting[i]
        hip = _make_divisible(hc / 2, 8)
        oup = _make_divisible(oc, 8)
        if i > 0:
            layers.append(MP(k=2))
        for i in range(num_block):
            if i != (num_block-1):
                new_oup = inp
            else:
                new_oup = oup
            layers.append(ELANBlock(inp, hip, new_oup, ln, ci, with_ibn=with_ibn))
            inp = new_oup
            
#        if i > 0:
#            layers.append(Non_local(oup, "BN"))

        return nn.Sequential(*layers), oup


    def forward(self, x):
#        x = self.features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.cell1([x])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # load pretrained weights
        if pretrained:
            checkpoint = torch.load(pretrained)
            if "model" in checkpoint.keys():
                self.load_state_dict(checkpoint['model'], strict=False)
            else:
                self.load_state_dict(checkpoint, strict=False)

@BACKBONE_REGISTRY.register()
def build_elan_nas_backbone(cfg):
    """
    Create a elan instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain      = cfg['MODEL']['BACKBONE']['PRETRAIN']
    pretrain_path = cfg['MODEL']['BACKBONE']['PRETRAIN_PATH']
    with_ibn      = cfg['MODEL']['BACKBONE']['WITH_IBN']
    feat_dim      = cfg['MODEL']['BACKBONE']['FEAT_DIM']
    # fmt: on

    model = ELANNas(with_ibn=with_ibn, feat_dim = feat_dim)

    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e

            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                logger.info(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                logger.info(
                    get_unexpected_parameters_message(incompatible.unexpected_keys)
                )

    return model
