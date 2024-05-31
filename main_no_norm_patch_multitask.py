import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.tueg_pretrain_conv import tueg_s4_pretrain_conv_patch_psd as tueg_s4_pretrain_conv
from src.data.dataset import TUEG_Dataset_chunked_psd as TUEG_Dataset
from src.engine.trainer import trainer_chunked_psd as trainer

import sys
import os
import logging
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_deterministic(seed: int=42):
    # Seed  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def cosine_similarity_loss(x, y):
    # Compute the cosine similarity
    cosine_sim = F.cosine_similarity(x, y, dim=2)
    
    # Calculate the loss as 1 - cosine similarity
    loss = torch.mean(1 - cosine_sim)
    
    return loss

def main():
    mask_ratio = float(sys.argv[1])
    is_mask = bool(int(sys.argv[2]))
    norm_data = bool(int(sys.argv[3]))
    
    n_layers_cnn = int(sys.argv[4])
    n_layers_s4 = int(sys.argv[5])
    
    seed = int(sys.argv[6])
    resume = bool(int(sys.argv[7]))
    resume_checkpoint = None
    if len(sys.argv) > 8:
        resume_checkpoint = sys.argv[8]

    batch_size = 32
    ratio_loss = 5
    # Set a random seed for reproducibility
    set_seed(seed)

    tueg_train_filepath = 'eeg_signals_chuncked_psd_train.txt'
    tueg_val_filepath = 'eeg_signals_chuncked_psd_val.txt'

    # Instantiate the model
    model = tueg_s4_pretrain_conv(device=device, ratio=mask_ratio, n_layers_cnn=n_layers_cnn,
                                  is_mask=is_mask, n_layers_s4=n_layers_s4).to(device)

    # Specify the optimizer and learning rate
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Specify the loss criterion
    loss_criterion_recon = cosine_similarity_loss
    loss_criterion_psd = nn.L1Loss()

    # Load your datasets and create data loaders
    train_dataset = TUEG_Dataset(tueg_train_filepath, n_windows=15360, norm_data=norm_data)
    val_dataset = TUEG_Dataset(tueg_val_filepath, n_windows=15360, n_files=1600, norm_data = norm_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Instantiate the trainer
    num_epochs = 100
    checkpoint_dir = '/scratch1/akommine/EMBC_SUBMISSION/checkpoints_psd_ncnn{}_ns4{}_rs{}/auto_mae_batchsize{}_s4_{}_norm{}_patch_{}mask_psd/'.format(n_layers_cnn, n_layers_s4, seed, batch_size, n_layers_s4, norm_data, mask_ratio)  # Set the directory where you want to save the model checkpoints and losses
    checkpoint_prefix = 'tueg_s4_{}_pretrain_conv'.format(mask_ratio)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logging.basicConfig(filename=checkpoint_dir + '/logfile', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    model_trainer = trainer(num_epochs, checkpoint_dir, checkpoint_prefix, is_mask=is_mask)

    # Train the model
    model_trainer.train(model, train_dataloader, val_dataloader,
                        optimizer, loss_criterion_recon, loss_criterion_psd, ratio_loss=ratio_loss,
                        resume=resume, resume_checkpoint=resume_checkpoint, device=device)

if __name__ == '__main__':
    main()
