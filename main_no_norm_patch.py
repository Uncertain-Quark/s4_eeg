import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.tueg_pretrain_conv import tueg_s4_pretrain_conv_patch as tueg_s4_pretrain_conv
from src.data.dataset import TUEG_Dataset_chunked as TUEG_Dataset
from src.engine.trainer import trainer_chunked as trainer

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

def set_seed(seed: int=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def cosine_similarity_loss(x, y):
    # Compute the cosine similarity
    cosine_sim = F.cosine_similarity(x, y, dim=-1)
    
    # Calculate the loss as 1 - cosine similarity
    loss = torch.mean(1 - cosine_sim)
    
    return loss

def main():
    mask_ratio = float(sys.argv[1])
    is_mask = bool(int(sys.argv[2]))
    n_layers_s4 = 8
    norm_data = bool(int(sys.argv[3]))
    resume = bool(int(sys.argv[4]))
    resume_checkpoint = None
    if len(sys.argv) > 5:
        resume_checkpoint = sys.argv[5]
    batch_size = 32
    # Set a random seed for reproducibility
    seed = 42
    set_seed(seed)

    tueg_train_filepath = 'eeg_signals_chunked_train.txt'
    tueg_val_filepath = 'eeg_signals_chunked_val.txt'

    # Instantiate the model
    model = tueg_s4_pretrain_conv(device=device, ratio=mask_ratio,
                                  is_mask=is_mask, n_layers_s4=n_layers_s4).to(device)

    # Specify the optimizer and learning rate
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Specify the loss criterion
    loss_criterion = cosine_similarity_loss

    # Load your datasets and create data loaders
    train_dataset = TUEG_Dataset(tueg_train_filepath, n_windows=15360, norm_data=norm_data)
    val_dataset = TUEG_Dataset(tueg_val_filepath, n_windows=15360, n_files=3200, norm_data = norm_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Instantiate the trainer
    num_epochs = 100
    checkpoint_dir = '/scratch2/akommine/NEAT_DATA/checkpoints_new/auto_mae_batchsize{}_s4_{}_norm{}_patch_{}mask/'.format(batch_size, n_layers_s4, norm_data, mask_ratio)  # Set the directory where you want to save the model checkpoints and losses
    checkpoint_prefix = 'tueg_s4_{}_pretrain_conv'.format(mask_ratio)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logging.basicConfig(filename=checkpoint_dir + '/logfile', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    model_trainer = trainer(num_epochs, checkpoint_dir, checkpoint_prefix, is_mask=is_mask)

    # Train the model
    model_trainer.train(model, train_dataloader, val_dataloader,
                        optimizer, loss_criterion,
                        resume=resume, resume_checkpoint=resume_checkpoint, device=device)

if __name__ == '__main__':
    main()
