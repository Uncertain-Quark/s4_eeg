# Author: Aditya
# Date: 2023-08

import torch
import torch.nn as nn
import torch.nn.functional as F

class block_conv(nn.Module):
    def __init__(self, in_channel: int=512, out_channel: int=512, 
                 kernel_size: int=3, stride: int=1, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU, conv: str='conv'):
        '''A single convolution block with normaliation and dropout'''
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding = padding
        self.output_padding = output_padding
        self.stride = stride
        self.dropout = dropout

        if residual is not False:
            raise NotImplementedError

        if conv == 'conv':
            self.conv_layer = nn.Conv1d(in_channels=self.in_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride, padding=self.padding)
        else:
            self.conv_layer = nn.ConvTranspose1d(in_channels=self.in_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride, padding=self.padding,
                                        output_padding=self.output_padding)
        
        self.norm_layer = norm(1, out_channel)

        self.dropout_layer = nn.Dropout1d(self.dropout)

        if act_layer is not None:
            self.act_layer = act_layer()
        else:
            self.act_layer = None

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.dropout_layer(x)
        x = self.norm_layer(x)

        if self.act_layer is not None:
            x = self.act_layer(x)
        return x

class block_last_decoder(block_conv):
    def __init__(self, in_channel: int = 512, out_channel: int = 512, kernel_size: int = 3, stride: int = 1, padding: int = 1, output_padding: int = 1, norm=nn.GroupNorm, dropout: float = 0.3, residual: bool = False, act_layer=nn.GELU, conv: str = 'conv'):
        super().__init__(in_channel, out_channel, kernel_size, stride, padding, output_padding, norm, dropout, residual, act_layer, conv)
    
    def forward(self, x):
        x = self.conv_layer(x)
        return x
    
class conv_encoder_tueg(nn.Module):
    def __init__(self, n_layers: int=4, input_channels: int=19, embedding_size: int=512, 
                 kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU):
        super().__init__()

        self.n_layers = n_layers

        self.block_layers = nn.ModuleList()

        self.block_layers.append(block_conv(input_channels, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer))
        
        for i in range(n_layers - 1):
            self.block_layers.append(block_conv(embedding_size, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer))

        
    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
        return x
    
class conv_decoder_tueg(nn.Module):
    def __init__(self, n_layers: int=4, input_channels: int=19, embedding_size: int=512, 
                 kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU):
        super().__init__()

        self.n_layers = n_layers

        self.block_layers = nn.ModuleList()
        
        for i in range(n_layers - 1):
            self.block_layers.append(block_conv(embedding_size, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer, 'deconv'))
        
        self.block_layers.append(block_last_decoder(embedding_size, input_channels,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, None, 'deconv'))
        
    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
        return x

if __name__ == '__main__':
    Fs, T = 250, 60

    encoder_sample = conv_encoder_tueg().to('cuda')
    decoder_sample = conv_decoder_tueg().to('cuda')

    encoder_sample_input = torch.randn(8, 19, 15360).to('cuda')

    encoder_sample_output = encoder_sample(encoder_sample_input)
    decoder_output = decoder_sample(encoder_sample_output)

    print(f'Output Shape of the SincConv : {encoder_sample_output.shape}')
    print(f'Shape of the input {encoder_sample_input.shape} Shape of the decoder output {decoder_output.shape}')
