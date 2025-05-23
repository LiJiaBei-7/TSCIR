from collections import OrderedDict
from typing import Tuple, Union
from typing import List, Optional
from torch import Tensor

import os
import json
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

import copy
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="gelu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn=False):
        
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        out = tgt + self.dropout2(tgt2)
        out = self.norm2(out)
        
        if return_attn:
            return out, attn
        else:
            return out

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=1, dropout=0.1):
        super().__init__()
        crossattn = CrossAttentionLayer(d_model, nhead, dropout)
        self.layers = _get_clones(crossattn, num_layers)

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn=False):
        output = tgt

        for layer in self.layers:
            if return_attn:
                output, attn = layer(output, memory, memory_key_padding_mask, pos, query_pos, return_attn)
            else:
                output = layer(output, memory, memory_key_padding_mask, pos, query_pos, return_attn)
        
        if return_attn:
            return output, attn 
        else: 
            return output
            
