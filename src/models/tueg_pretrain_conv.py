from .encoder_conv import conv_encoder_tueg
from .encoder_conv import conv_decoder_tueg
from .s4_model import S4Model
from .multitask import psd_projection

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import random

class patch_masking(nn.Module):
    def __init__(self, ratio: float=0.3, patch_size: float=0.05, device: str='cuda'):
        super().__init__()

        self.ratio = ratio
        self.patch_size = patch_size
        self.device = device

        self.num_patches_masked = int(round(self.ratio/self.patch_size))

        self.patch_indices = np.linspace(0, 1, int(round(1/self.patch_size)), endpoint=False)
    
    def __generate_mask__(self, shape_mask):
        timestamps = shape_mask[-1]
        mask = torch.ones(shape_mask)

        if self.ratio == 0.0:
            return mask.to(self.device)
        
        # Generate the patches to mask
        indices_mask = np.random.choice(self.patch_indices, self.num_patches_masked, replace=False)

        for index_mask in indices_mask:
            start_mask, end_mask = int(timestamps*index_mask), int(timestamps*(index_mask + self.patch_size))
            mask[:, :, start_mask:end_mask] = 0.0 
        return mask.to(self.device)

    def forward(self, x):
        mask = self.__generate_mask__(x.shape)
        masked_x = mask * x
        return mask, masked_x

class tueg_s4_pretrain_conv(nn.Module):
    def __init__(self, n_layers_cnn: int=6
                 , n_layers_s4: int=8
                 , Fs: int=250, ratio: float=0.2,
                 device: str='cuda', embedding_size: int=512,
                 is_mask: bool=True):
        super().__init__()

        self.n_layers_cnn = n_layers_cnn
        self.n_layers_s4 = n_layers_s4
        self.Fs = Fs
        self.device = device
        self.is_mask = is_mask

        self.conv_encoder = conv_encoder_tueg(n_layers_cnn, embedding_size=embedding_size)
        self.deconv_encoder = conv_decoder_tueg(n_layers_cnn, embedding_size=embedding_size)

        self.s4_model = S4Model(
        d_model=embedding_size,
        n_layers=n_layers_s4,
        dropout=0.3,
        prenorm=False)

        self.mask = random_masking(ratio)
    
    def forward(self, x):
        x = x.to(self.device)

        mask = None
        # Mask the input
        if self.is_mask:
            mask = self.mask(x.clone()).to(self.device)
            masked_input = mask * x
        else: 
            masked_input = x

        conv_output = self.conv_encoder(masked_input)

        s4_output = self.s4_model(conv_output.transpose(-1, -2))

        decoder_out = self.deconv_encoder(s4_output.transpose(-1, -2))

        return x, mask, decoder_out

class tueg_s4_pretrain_conv_patch(nn.Module):
    def __init__(self, n_layers_cnn: int=6
                 , n_layers_s4: int=8
                 , Fs: int=250, ratio: float=0.2,
                 device: str='cuda', embedding_size: int=512,
                 is_mask: bool=True, in_channels: int=19):
        super().__init__()

        self.n_layers_cnn = n_layers_cnn
        self.n_layers_s4 = n_layers_s4
        self.Fs = Fs
        self.device = device
        self.is_mask = is_mask

        self.conv_encoder = conv_encoder_tueg(n_layers_cnn, embedding_size=embedding_size, input_channels=in_channels)
        self.deconv_encoder = conv_decoder_tueg(n_layers_cnn, embedding_size=embedding_size, input_channels=in_channels)

        self.s4_model = S4Model(d_input=embedding_size,
        d_output=embedding_size,
        d_model=embedding_size,
        n_layers=n_layers_s4,
        dropout=0.3,
        prenorm=False)

        self.mask = patch_masking(ratio)
    
    def forward(self, x):
        x = x.to(self.device)

        mask = None
        # Mask the input
        if self.is_mask:
            mask, masked_input = self.mask(x)
        else: 
            masked_input = x

        conv_output = self.conv_encoder(masked_input.clone())

        s4_output = self.s4_model(conv_output.transpose(-1, -2).clone())

        decoder_out = self.deconv_encoder(s4_output.transpose(-1, -2).clone())

        return x, mask, decoder_out

class tueg_s4_pretrain_conv_patch_psd(tueg_s4_pretrain_conv_patch):
    def __init__(self, n_layers_cnn: int = 6, n_layers_s4: int = 8, Fs: int = 250, ratio: float = 0.2, device: str = 'cuda', embedding_size: int = 512, is_mask: bool = True, in_channels: int = 19):
        super().__init__(n_layers_cnn, n_layers_s4, Fs, ratio, device, embedding_size, is_mask, in_channels)
        self.projector = psd_projection()
    
    def forward(self, x):
        x = x.to(self.device)

        mask = None
        # Mask the input
        if self.is_mask:
            mask, masked_input = self.mask(x)
        else: 
            masked_input = x

        conv_output = self.conv_encoder(masked_input.clone())

        s4_output = self.s4_model(conv_output.transpose(-1, -2).clone())

        psd = self.projector(s4_output.transpose(-1, -2).clone())

        decoder_out = self.deconv_encoder(s4_output.transpose(-1, -2).clone())

        return x, mask, decoder_out, psd
    
class tueg_s4_pretrain_conv_patch_dual(nn.Module):
    def __init__(self, n_layers_cnn: int=6
                 , n_layers_s4: int=8
                 , Fs: int=250, ratio: float=0.2,
                 device: str='cuda', embedding_size: int=512,
                 is_mask: bool=True):
        super().__init__()

        self.n_layers_cnn = n_layers_cnn
        self.n_layers_s4 = n_layers_s4
        self.Fs = Fs
        self.device = device
        self.is_mask = is_mask

        self.conv_encoder = conv_encoder_tueg(n_layers_cnn, embedding_size=embedding_size)
        self.deconv_encoder = conv_decoder_tueg(n_layers_cnn, embedding_size=embedding_size)

        self.s4_model = S4Model(d_input=embedding_size,
        d_output=embedding_size,
        d_model=embedding_size,
        n_layers=n_layers_s4,
        dropout=0.3,
        prenorm=False)

        self.mask = patch_masking(ratio)
    
    def forward(self, x):
        x = x.to(self.device)

        conv_output = self.conv_encoder(x.clone())

        mask = None
        # Mask the input
        if self.is_mask:
            mask, masked_input = self.mask(conv_output)
        else: 
            masked_input = conv_output

        s4_output = self.s4_model(masked_input.transpose(-1, -2).clone())

        decoder_out = self.deconv_encoder(s4_output.transpose(-1, -2).clone())

        return x, conv_output, s4_output.transpose(-1, -2), decoder_out, mask


if __name__ == '__main__':

    tueg_s4_pretrain_obj = tueg_s4_pretrain_conv_patch().to('cuda')

    sample_input = torch.randn(32, 19, 15360).to('cuda')
    conv_output = tueg_s4_pretrain_obj.conv_encoder(sample_input)

    print(conv_output.shape)

