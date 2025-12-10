# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging


import torch
import torch.nn as nn
from torch.nn import Dropout, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import numpy as np


from .encoder_decoder import unet_TSB_encoder, unet_TSB_decoder
from .Transformer_utils import AbsolutePositionalEncoder, Block, np2th
from .conformer import TSCB



logger = logging.getLogger(__name__)




class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self):
        super(Embeddings, self).__init__()
        # patch_size = (1, 1)  # patch size equals to a pixel of the feature map
        self.hybrid_model = unet_TSB_encoder(ngf=128, input_nc=1)
        # output channels of the self.hybrid_model * downsample ratio
        in_channels = self.hybrid_model.ngf * self.hybrid_model.time_downsample_ratio

        # in_channels = 128 * 2^3 
        # self.patch_embeddings = Conv2d(in_channels=in_channels,
        #                                out_channels=hidden_size,
        #                                kernel_size=patch_size,
        #                                stride=patch_size)

        # self.dropout = Dropout(dropout_rate)


    def forward(self, x):
        x, features, origin_len, pad_input = self.hybrid_model(x)
        # x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        # x = x.flatten(2)
        # x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # b, n, hidden = x.shape


        # pos_embedding = AbsolutePositionalEncoder(hidden)
        # embeddings = x + pos_embedding(x)
        # embeddings = self.dropout(embeddings)

        return x, features, origin_len, pad_input



class Encoder(nn.Module):
    def __init__(self, 
                 inchannel, 
                 conformer_num_layers, 
                 ):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(conformer_num_layers):
            layer = TSCB(inchannel)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
            
        return hidden_states


class Transformer(nn.Module):
    def __init__(self, 
                 inchannel,
                 conformer_num_layers):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings()
        self.encoder = Encoder(inchannel, conformer_num_layers)

    def forward(self, input_ids):
        embedding_output, features, origin_len, pad_input = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, origin_len, pad_input




class ConformerUnet(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 inchannel,
                 conformer_num_layers):
        super(ConformerUnet, self).__init__()
        self.transformer = Transformer(inchannel, conformer_num_layers)
        self.decoder = unet_TSB_decoder(hidden_size)

    def forward(self, input):
        x, features, origin_len, pad_input = self.transformer(input)  # (B, n_patch, hidden)
        output = self.decoder(x, features, origin_len, pad_input)
        return output



if __name__ == "__main__":                
    a = torch.randn(60,1,80,80)
    model = ConformerUnet(1024,1024,3)
    b= model(a)
    print(b.shape)
    