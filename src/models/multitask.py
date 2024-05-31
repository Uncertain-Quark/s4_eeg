import torch
import torch.nn as nn
import torch.nn.functional as F

class psd_projection(nn.Module):
    def __init__(self, dim_in: int=512, dim_out: int=95):
        self.dim_in = dim_in
        self.dim_out = dim_out
        super().__init__()

        self.time_dim = 16

        # Declare the projections
        self.fc_layer = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        # Shape of X: (batch, channels=512, 16, 15)
        x = x.view(x.shape[0], x.shape[1], self.time_dim, int(x.shape[2]/self.time_dim))
        # Shape of X: (batch, channels=512, 16)
        x = F.gelu(x.mean(dim=3))
        # Shape of X: (batch, 16, channels=512)
        x = x.transpose(-1, -2)
        # Shape of X: (batch, 16, 95)
        x = self.fc_layer(x)

        return x

if __name__ == '__main__':
    sample_tensor = torch.randn(32, 512, 240)
    sample_projection = psd_projection()

    sample_output = sample_projection(sample_tensor)

    print(f'Shape of the input {sample_tensor.shape} Shape of the output: {sample_output.shape}')

