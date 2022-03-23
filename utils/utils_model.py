# -*- coding: utf-8 -*-
import numpy as np
import torch
from utils import utils_image as util


'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''


def test_mode(model, L, mode=0, refield=128, min_size=256, sf=1, modulo=1):
    '''
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Some testing modes
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # (0) normal: test(model, L)
    # (1) pad: test_pad(model, L, modulo=16)
    # (2) split: test_split(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (3) x8: test_x8(model, L, modulo=1)
    # (4) split and x8: test_split_x8(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (5) split only once: test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # ---------------------------------------
    '''
    if mode == 0:
        E = test(model, L)
    elif mode == 1:
        E = test_pad(model, L, modulo)
    elif mode == 2:
        E = test_split(model, L, refield, min_size, sf, modulo)
    elif mode == 3:
        E = test_x8(model, L, modulo)
    elif mode == 4:
        E = test_split_x8(model, L, refield, min_size, sf, modulo)
    elif mode == 5:
        E = test_onesplit(model, L, refield, min_size, sf, modulo)
    return E


'''
# ---------------------------------------
# normal (0)
# ---------------------------------------
'''


def test(model, L):
    E = model(L)
    return E


'''
# ---------------------------------------
# pad (1)
# ---------------------------------------
'''


def test_pad(model, L, modulo=16):
    h, w = L.size()[-2:]
    paddingBottom = int(np.ceil(h/modulo)*modulo-h)
    paddingRight = int(np.ceil(w/modulo)*modulo-w)
    L = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(L)
    E = model(L)
    E = E[..., :h, :w]
    return E


'''
# ---------------------------------------
# split (function)
# ---------------------------------------
'''


def test_split_fn(model, L, refield=32, min_size=256, sf=1, modulo=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]
    if h*w <= min_size**2:
        L = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(L)
        E = model(L)
        E = E[..., :h*sf, :w*sf]
    else:
        top = slice(0, (h//2//refield+1)*refield)
        bottom = slice(h - (h//2//refield+1)*refield, h)
        left = slice(0, (w//2//refield+1)*refield)
        right = slice(w - (w//2//refield+1)*refield, w)
        Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]

        if h * w <= 4*(min_size**2):
            Es = [model(Ls[i]) for i in range(4)]
        else:
            Es = [test_split_fn(model, Ls[i], refield=refield, min_size=min_size, sf=sf, modulo=modulo) for i in range(4)]

        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

        E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
        E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
        E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
        E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E



def test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]

    top = slice(0, (h//2//refield+1)*refield)
    bottom = slice(h - (h//2//refield+1)*refield, h)
    left = slice(0, (w//2//refield+1)*refield)
    right = slice(w - (w//2//refield+1)*refield, w)
    Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
    Es = [model(Ls[i]) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
    E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
    E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
    E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E



'''
# ---------------------------------------
# split (2)
# ---------------------------------------
'''


def test_split(model, L, refield=32, min_size=256, sf=1, modulo=1):
    E = test_split_fn(model, L, refield=refield, min_size=min_size, sf=sf, modulo=modulo)
    return E


'''
# ---------------------------------------
# x8 (3)
# ---------------------------------------
'''


def test_x8(model, L, modulo=1):
    E_list = [test_pad(model, util.augment_img_tensor(L, mode=i), modulo=modulo) for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = util.augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = util.augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E


'''
# ---------------------------------------
# split and x8 (4)
# ---------------------------------------
'''


def test_split_x8(model, L, refield=32, min_size=256, sf=1, modulo=1):
    E_list = [test_split_fn(model, util.augment_img_tensor(L, mode=i), refield=refield, min_size=min_size, sf=sf, modulo=modulo) for i in range(8)]
    for k, i in enumerate(range(len(E_list))):
        if i==3 or i==5:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=8-i)
        else:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E


'''
# ^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^
# _^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_
# ^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^
'''


'''
# ---------------------------------------
# print
# ---------------------------------------
'''


# -------------------
# print model
# -------------------
def print_model(model):
    msg = describe_model(model)
    print(msg)


# -------------------
# print params
# -------------------
def print_params(model):
    msg = describe_params(model)
    print(msg)


'''
# ---------------------------------------
# information
# ---------------------------------------
'''


# -------------------
# model inforation
# -------------------
def info_model(model):
    msg = describe_model(model)
    return msg


# -------------------
# params inforation
# -------------------
def info_params(model):
    msg = describe_params(model)
    return msg


'''
# ---------------------------------------
# description
# ---------------------------------------
'''


# ----------------------------------------------
# model name and total number of parameters
# ----------------------------------------------
def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


# ----------------------------------------------
# parameters description
# ----------------------------------------------
def describe_params(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'param_name') + '\n'
    for name, param in model.state_dict().items():
        if not 'num_batches_tracked' in name:
            v = param.data.clone().float()
            msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), name) + '\n'
    return msg


if __name__ == '__main__':

    class Net(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model = Net()
    model = model.eval()
    print_model(model)
    print_params(model)
    x = torch.randn((2,3,400,400))
    torch.cuda.empty_cache()
    with torch.no_grad():
        for mode in range(5):
            y = test_mode(model, x, mode)
            print(y.shape)
