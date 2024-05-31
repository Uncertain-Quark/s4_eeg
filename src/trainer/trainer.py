import os
import torch
import pickle
import glob
import logging
import torch.nn as nn
import torch.nn.functional as F

class trainer:
    def __init__(self, n_epochs, checkpoint_dir, checkpoint_prefix, is_mask: bool=True):
        self.n_epochs = n_epochs
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.is_mask = is_mask
    
    def validate(self, model, val_dataloader, loss_criterion, device='cuda'):
        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for val_eeg_data in val_dataloader:
                val_eeg_data = val_eeg_data.to(device)
                val_encoder_masked_output, val_decoder_masked_output = self.masked_loss_forward(model, val_eeg_data)

                val_loss = loss_criterion(val_decoder_masked_output, val_encoder_masked_output)

                total_val_loss += val_loss.item() * val_eeg_data.size(0)
                total_val_samples += val_eeg_data.size(0)

        avg_val_loss = total_val_loss / total_val_samples
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    
    def masked_loss_forward(self, model, eeg_data):
        encoder_output, mask, decoder_output = model(eeg_data)
        if self.is_mask:
            #TODO take only the masked parts and learn on them
            # encoder_masked_output = (1 - mask) * encoder_output
            # decoder_masked_output = (1 - mask) * decoder_output
            return encoder_output, decoder_output
        else:
            return encoder_output, decoder_output
    
    def resume_checkpoint(self, resume, resume_checkpoint, model):
        if resume and resume_checkpoint is not None:
            resume_checkpoint_path = os.path.join(self.checkpoint_dir, resume_checkpoint)
            if os.path.exists(resume_checkpoint_path):
                logging.info(f"Resuming from checkpoint: {resume_checkpoint_path}")
                model.load_state_dict(torch.load(resume_checkpoint_path))
                checkpoint_parts = os.path.basename(resume_checkpoint_path).split('_')
                self.start_epoch = int(checkpoint_parts[-1].split('.')[0]) + 1
            else:
                logging.info(f"Warning: Could not find resume checkpoint: {resume_checkpoint_path}")
                logging.info("Training from scratch...")
        return model

    def train(self, model, train_dataloader, val_dataloader,
              optimizer, loss_criterion,
              resume: bool = False, resume_checkpoint: str = None,
              device: str = 'cuda'):

        model = model.to(device)
        model = self.resume_checkpoint(resume, resume_checkpoint, model)

        for epoch in range(self.start_epoch, self.n_epochs):
            logging.info(f"Epoch {epoch+1}/{self.n_epochs}")

            model.train()
            total_loss = 0.0
            total_samples = 0

            for batch_id, eeg_data in enumerate(train_dataloader):
                eeg_data = eeg_data.to(device)
                optimizer.zero_grad()

                encoder_masked_output, decoder_masked_output = self.masked_loss_forward(model, eeg_data)
                loss = loss_criterion(decoder_masked_output, encoder_masked_output)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * eeg_data.size(0)
                total_samples += eeg_data.size(0)

                logging.info(f'Epoch {epoch} Loss in the batch {batch_id} {loss.item()}')

            avg_loss = total_loss / total_samples
            logging.info(f"Train Loss: {avg_loss:.4f}")

            self.train_losses.append(avg_loss)

            avg_val_loss = self.validate(model, val_dataloader, loss_criterion, device=device)
            self.val_losses.append(avg_val_loss)

            checkpoint_filename = f"{self.checkpoint_prefix}_epoch_{epoch+1}.pt"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_path)

            losses_filename = f"{self.checkpoint_prefix}_epoch_{epoch+1}_losses.pkl"
            losses_path = os.path.join(self.checkpoint_dir, losses_filename)
            with open(losses_path, 'wb') as file:
                pickle.dump({'train_losses': self.train_losses, 'val_losses': self.val_losses}, file)

        print("Training completed!")

class trainer_chunked(trainer):
    def __init__(self, n_epochs, checkpoint_dir, checkpoint_prefix, is_mask: bool = True):
        super().__init__(n_epochs, checkpoint_dir, checkpoint_prefix, is_mask)
        self.batch_counter = 0
        self.batch_loss = 0.0
        self.batch_loss_list = []

    def resume_checkpoint(self, resume, resume_checkpoint, model):
        if resume and resume_checkpoint is not None:
            resume_checkpoint_path = os.path.join(self.checkpoint_dir, resume_checkpoint)
            if os.path.exists(resume_checkpoint_path):
                logging.info(f"Resuming from checkpoint: {resume_checkpoint_path}")
                model.load_state_dict(torch.load(resume_checkpoint_path))
                checkpoint_parts = os.path.basename(resume_checkpoint_path).split('_')
                self.batch_counter = int(checkpoint_parts[-4])
                
                batch_losses_filename = f"{self.checkpoint_prefix}_batches_{self.batch_counter - 1999}_{self.batch_counter}_batch_losses.pkl"
                batch_losses_path = os.path.join(self.checkpoint_dir, batch_losses_filename)
                with open(batch_losses_path, 'rb') as file:
                    data_dict = pickle.load(file)
                self.batch_loss_list = data_dict['train_losses']

                epoch_checkpoint_list = glob.glob(self.checkpoint_dir + '/' + self.checkpoint_prefix + '*epoch*losses.pkl')
                epoch_checkpoint_list.sort()

                epoch_checkpoint = epoch_checkpoint_list[-1].split('_')
                self.start_epoch = int(epoch_checkpoint[-2])

                with open(epoch_checkpoint_list[-1], 'rb') as f:
                    data_dict = pickle.load(f)
                
                self.train_losses = data_dict['train_losses']
                self.val_losses = data_dict['val_losses']
            else:
                print(f"Warning: Could not find resume checkpoint: {resume_checkpoint_path}")
                print("Training from scratch...")
        return model
    
    def validate(self, model, val_dataloader, loss_criterion, device='cuda'):
        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for val_batch_id, val_data in enumerate(val_dataloader):
                val_data = val_data.to(device)

                encoder_masked_output, decoder_masked_output = self.masked_loss_forward(model, val_data)
                loss = loss_criterion(decoder_masked_output, encoder_masked_output)

                total_loss += loss.item() * val_data.size(0)
                total_samples += val_data.size(0)

        avg_val_loss = total_loss / total_samples
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss
    
    def validate_and_save_losses(self, model, val_dataloader, loss_criterion, device='cuda'):
        checkpoint_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pt')))

        val_losses = {}
        for checkpoint_path in checkpoint_files:
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            
            model.load_state_dict(torch.load(checkpoint_path))

            avg_val_loss = self.validate(model, val_dataloader, loss_criterion, device=device)
            val_losses[checkpoint_path] = avg_val_loss

            # Save the validation loss to a file
            checkpoint_name = os.path.basename(checkpoint_path)
            loss_filename = os.path.splitext(checkpoint_name)[0] + '_val_loss.pkl'
            loss_filepath = os.path.join(self.checkpoint_dir, loss_filename)
            val_dict = {'val_losses': avg_val_loss}
            with open(loss_filepath, 'wb') as file:
                pickle.dump(val_dict, file)
        
        with open(os.path.join(self.checkpoint_dir, 'val_losses.pkl'), 'wb') as f:
            pickle.dump(val_losses, f)


    def train(self, model, train_dataloader, val_dataloader,
              optimizer, loss_criterion,
              resume: bool = False, resume_checkpoint: str = None,
              device: str = 'cuda'):

        model = model.to(device)
        model = self.resume_checkpoint(resume, resume_checkpoint, model)

        for epoch in range(self.start_epoch, self.n_epochs):
            logging.info(f"Epoch {epoch + 1}/{self.n_epochs}")

            model.train()
            total_loss = 0.0
            total_samples = 0

            for batch_id, eeg_data in enumerate(train_dataloader):
                self.batch_counter += 1

                eeg_data = eeg_data.to(device)
                optimizer.zero_grad()

                encoder_masked_output, _, decoder_masked_output = model(eeg_data)
                loss = loss_criterion(decoder_masked_output, encoder_masked_output)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * eeg_data.size(0)
                total_samples += eeg_data.size(0)
                self.batch_loss += loss.item()

                logging.info(f'Epoch {epoch} Loss in batch {batch_id}: {loss.item()}')

                if self.batch_counter % 2000 == 0:
                    avg_batch_loss = self.batch_loss / 2000
                    self.batch_loss_list.append(avg_batch_loss)
                    self.batch_loss = 0.0

                    # Save the batch loss list
                    batch_number = self.batch_counter
                    batch_losses_filename = f"{self.checkpoint_prefix}_batches_{batch_number - 1999}_{batch_number}_batch_losses.pkl"
                    batch_losses_path = os.path.join(self.checkpoint_dir, batch_losses_filename)
                    with open(batch_losses_path, 'wb') as file:
                        pickle.dump({'train_losses': self.batch_loss_list}, file)

                    # Save the model checkpoint
                    model_checkpoint_filename = f"{self.checkpoint_prefix}_batches_{batch_number - 1999}_{batch_number}_avg_loss_{avg_batch_loss:.4f}.pt"
                    model_checkpoint_path = os.path.join(self.checkpoint_dir, model_checkpoint_filename)
                    
                    # Assuming you have a PyTorch model named 'model' that you want to save
                    torch.save(model.state_dict(), model_checkpoint_path)

            avg_loss = total_loss / total_samples
            logging.info(f"Train Loss: {avg_loss:.4f}")

            self.train_losses.append(avg_loss)

            avg_val_loss = self.validate(model, val_dataloader, loss_criterion, device=device)
            self.val_losses.append(avg_val_loss)

            # Save the model and loss list every epoch
            model_filename = f"{self.checkpoint_prefix}_epoch_{epoch + 1}.pt"
            model_path = os.path.join(self.checkpoint_dir, model_filename)
            torch.save(model.state_dict(), model_path)

            losses_filename = f"{self.checkpoint_prefix}_epoch_{epoch + 1}_losses.pkl"
            losses_path = os.path.join(self.checkpoint_dir, losses_filename)
            with open(losses_path, 'wb') as file:
                pickle.dump({'train_losses': self.train_losses, 'val_losses': self.val_losses}, file)

        print("Training completed!")
        
class trainer_chunked_psd(trainer_chunked):
    def __init__(self, n_epochs, checkpoint_dir, checkpoint_prefix, is_mask: bool = True):
        super().__init__(n_epochs, checkpoint_dir, checkpoint_prefix, is_mask)
        self.batch_counter = 0.0
        self.batch_loss = 0.0
        self.batch_loss_recon = 0.0
        self.batch_loss_psd = 0.0
        self.batch_loss_list = []
        self.batch_loss_list_psd = []
        self.batch_loss_list_recon = []
    
    def resume_checkpoint(self, resume, resume_checkpoint, model):
        if resume and resume_checkpoint is not None:
            resume_checkpoint_path = os.path.join(self.checkpoint_dir, resume_checkpoint)
            if os.path.exists(resume_checkpoint_path):
                logging.info(f"Resuming from checkpoint: {resume_checkpoint_path}")
                model.load_state_dict(torch.load(resume_checkpoint_path))
                checkpoint_parts = os.path.basename(resume_checkpoint_path).split('_')
                self.batch_counter = float(checkpoint_parts[-4])
                
                batch_losses_filename = f"{self.checkpoint_prefix}_batches_{self.batch_counter - 1999}_{self.batch_counter}_batch_losses.pkl"
                batch_losses_path = os.path.join(self.checkpoint_dir, batch_losses_filename)
                with open(batch_losses_path, 'rb') as file:
                    data_dict = pickle.load(file)
                self.batch_loss_list = data_dict['train_losses']
                self.batch_loss_list_recon = data_dict['train_losses_recon']
                self.batch_loss_list_psd = data_dict['train_losses_psd']

                epoch_checkpoint_list = glob.glob(self.checkpoint_dir + '/' + self.checkpoint_prefix + '*epoch*losses.pkl')
                epoch_checkpoint_list.sort()

                epoch_checkpoint = epoch_checkpoint_list[-1].split('_')
                self.start_epoch = int(epoch_checkpoint[-2])

                with open(epoch_checkpoint_list[-1], 'rb') as f:
                    data_dict = pickle.load(f)
                
                self.train_losses = data_dict['train_losses']
                self.val_losses = data_dict['val_losses']
            else:
                print(f"Warning: Could not find resume checkpoint: {resume_checkpoint_path}")
                print("Training from scratch...")
        return model
    
    def validate_and_save_losses(self, model, val_dataloader, loss_criterion_recon, loss_criterion_psd, ratio_loss, device='cuda'):
        checkpoint_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pt')))

        val_losses = {}
        for checkpoint_path in checkpoint_files:
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            
            model.load_state_dict(torch.load(checkpoint_path))

            avg_val_loss, avg_val_loss_recon, avg_val_loss_psd = self.validate(model, val_dataloader, loss_criterion_recon, loss_criterion_psd, ratio_loss, device=device)

            # Save the validation loss to a file
            checkpoint_name = os.path.basename(checkpoint_path)
            loss_filename = os.path.splitext(checkpoint_name)[0] + '_val_loss.pkl'
            loss_filepath = os.path.join(self.checkpoint_dir, loss_filename)
            val_dict = {'val_losses': avg_val_loss, 
                        'val_losses_recon': avg_val_loss_recon,
                          'val_losses_psd': avg_val_loss_psd}
            
            val_losses[checkpoint_path] = val_dict
            with open(loss_filepath, 'wb') as file:
                pickle.dump(val_dict, file)
        
        with open(os.path.join(self.checkpoint_dir, 'val_losses.pkl'), 'wb') as f:
            pickle.dump(val_losses, f)
    
    def validate(self, model, val_dataloader, loss_criterion_recon, loss_criterion_psd, ratio_loss, device='cuda'):
        model.eval()
        total_loss = 0.0
        total_loss_recon = 0.0
        total_loss_psd = 0.0
        total_samples = 0

        with torch.no_grad():
            for val_batch_id, val_data in enumerate(val_dataloader):
                val_eeg_data, val_psd = val_data
                val_eeg_data = val_eeg_data.to(device)
                val_psd = val_psd.to(device)

                encoder_masked_output, _, decoder_masked_output, psd_estimated = model(val_eeg_data.clone())
                loss_recon = loss_criterion_recon(decoder_masked_output, encoder_masked_output)
                loss_psd = loss_criterion_psd(psd_estimated, val_psd)
                loss = loss_recon + ratio_loss * loss_psd

                total_loss += loss.item() * val_eeg_data.size(0)
                total_loss_recon += loss_recon.item() * val_eeg_data.size(0)
                total_loss_psd += loss_psd.item() * val_eeg_data.size(0)
                logging.info(val_batch_id)
                total_samples += val_eeg_data.size(0)

        avg_val_loss = total_loss / total_samples
        avg_val_loss_recon = total_loss_recon / total_samples
        avg_val_loss_psd = total_loss_psd / total_samples
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss, avg_val_loss_recon, avg_val_loss_psd
    
    def train(self, model, train_dataloader, val_dataloader,
              optimizer, loss_criterion_recon, loss_criterion_psd, ratio_loss,
              resume: bool = False, resume_checkpoint: str = None,
              device: str = 'cuda'):

        model = model.to(device)
        model = self.resume_checkpoint(resume, resume_checkpoint, model)

        for epoch in range(self.start_epoch, self.n_epochs):
            logging.info(f"Epoch {epoch + 1}/{self.n_epochs}")

            model.train()
            total_loss = 0.0
            total_samples = 0

            for batch_id, data in enumerate(train_dataloader):
                eeg_data, psd = data
                psd = psd.to(device)

                self.batch_counter += 1

                eeg_data = eeg_data.to(device)
                optimizer.zero_grad()

                encoder_masked_output, _, decoder_masked_output, psd_estimated = model(eeg_data.clone())
                loss_recon = loss_criterion_recon(decoder_masked_output, eeg_data)
                loss_psd = loss_criterion_psd(psd_estimated, psd)
                loss = loss_recon + ratio_loss * loss_psd

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * eeg_data.size(0)
                total_samples += eeg_data.size(0)
                self.batch_loss += loss.item()
                self.batch_loss_recon = loss_recon.item()
                self.batch_loss_psd = loss_psd.item()

                logging.info(f'Epoch {epoch} Loss in batch {batch_id}: {loss.item()}')
                logging.info(f'Epoch {epoch} Recon Loss in batch {batch_id}: {loss_recon.item()}')
                logging.info(f'Epoch {epoch} PSD Loss in batch {batch_id}: {loss_psd.item()}')

                if self.batch_counter % 2000 == 0:
                    avg_batch_loss = self.batch_loss / 2000
                    avg_batch_loss_recon = self.batch_loss_recon / 2000
                    avg_batch_loss_psd = self.batch_loss_psd / 2000
                    
                    self.batch_loss_list.append(avg_batch_loss)
                    self.batch_loss_list_recon.append(avg_batch_loss_recon)
                    self.batch_loss_list_psd.append(avg_batch_loss_psd)

                    self.batch_loss = 0.0
                    self.batch_loss_recon = 0.0
                    self.batch_loss_psd = 0.0

                    # Save the batch loss list
                    batch_number = self.batch_counter
                    batch_losses_filename = f"{self.checkpoint_prefix}_batches_{batch_number - 1999}_{batch_number}_batch_losses.pkl"
                    batch_losses_path = os.path.join(self.checkpoint_dir, batch_losses_filename)
                    with open(batch_losses_path, 'wb') as file:
                        pickle.dump({'train_losses': self.batch_loss_list,
                                     'train_losses_recon': self.batch_loss_list_recon,
                                     'train_losses_psd': self.batch_loss_list_psd}, file)
                    
                    # Save the model checkpoint
                    model_checkpoint_filename = f"{self.checkpoint_prefix}_batches_{batch_number - 1999}_{batch_number}_avg_loss_{avg_batch_loss:.4f}.pt"
                    model_checkpoint_path = os.path.join(self.checkpoint_dir, model_checkpoint_filename)
                    
                    # Assuming you have a PyTorch model named 'model' that you want to save
                    torch.save(model.state_dict(), model_checkpoint_path)

            avg_loss = total_loss / total_samples
            logging.info(f"Train Loss: {avg_loss:.4f}")

            self.train_losses.append(avg_loss)

            avg_val_loss, _, _ = self.validate(model, val_dataloader, loss_criterion_recon, loss_criterion_psd, ratio_loss, device=device)
            self.val_losses.append(avg_val_loss)

            # Save the model and loss list every epoch
            model_filename = f"{self.checkpoint_prefix}_epoch_{epoch + 1}.pt"
            model_path = os.path.join(self.checkpoint_dir, model_filename)
            torch.save(model.state_dict(), model_path)

            losses_filename = f"{self.checkpoint_prefix}_epoch_{epoch + 1}_losses.pkl"
            losses_path = os.path.join(self.checkpoint_dir, losses_filename)
            with open(losses_path, 'wb') as file:
                pickle.dump({'train_losses': self.train_losses, 'val_losses': self.val_losses}, file)

        print("Training completed!")

class trainer_chunked_dual(trainer_chunked):
    def __init__(self, n_epochs, checkpoint_dir, checkpoint_prefix, is_mask: bool = True):
        super().__init__(n_epochs, checkpoint_dir, checkpoint_prefix, is_mask)
        self.batch_counter = 0
        self.batch_loss = 0.0
        self.batch_loss_recon = 0.0
        self.batch_loss_psd = 0.0
        self.batch_loss_list = []
        self.batch_loss_list_psd = []
        self.batch_loss_list_recon = []
    
    def resume_checkpoint(self, resume, resume_checkpoint, model):
        if resume and resume_checkpoint is not None:
            resume_checkpoint_path = os.path.join(self.checkpoint_dir, resume_checkpoint)
            if os.path.exists(resume_checkpoint_path):
                logging.info(f"Resuming from checkpoint: {resume_checkpoint_path}")
                model.load_state_dict(torch.load(resume_checkpoint_path))
                checkpoint_parts = os.path.basename(resume_checkpoint_path).split('_')
                self.batch_counter = int(checkpoint_parts[-4])
                
                batch_losses_filename = f"{self.checkpoint_prefix}_batches_{self.batch_counter - 1999}_{self.batch_counter}_losses.pkl"
                batch_losses_path = os.path.join(self.checkpoint_dir, batch_losses_filename)
                with open(batch_losses_path, 'rb') as file:
                    data_dict = pickle.load(file)
                self.batch_loss_list = data_dict['train_losses']
                self.batch_loss_list_recon = data_dict['train_losses_recon']
                self.batch_loss_list_psd = data_dict['train_losses_psd']

                epoch_checkpoint_list = glob.glob(self.checkpoint_dir + '/' + self.checkpoint_prefix + '*epoch*losses.pkl')
                epoch_checkpoint_list.sort()

                epoch_checkpoint = epoch_checkpoint_list[-1].split('_')
                self.start_epoch = int(epoch_checkpoint[-2])

                with open(epoch_checkpoint_list[-1], 'rb') as f:
                    data_dict = pickle.load(f)
                
                self.train_losses = data_dict['train_losses']
                self.val_losses = data_dict['val_losses']
            else:
                print(f"Warning: Could not find resume checkpoint: {resume_checkpoint_path}")
                print("Training from scratch...")
        return model
    
    def validate_and_save_losses(self, model, val_dataloader, loss_criterion_recon, loss_criterion_embed, ratio_loss, device='cuda'):
        checkpoint_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pt')))

        val_losses = {}
        for checkpoint_path in checkpoint_files:
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            
            model.load_state_dict(torch.load(checkpoint_path))

            avg_val_loss, avg_val_loss_embed, avg_val_loss_signal = self.validate(model, val_dataloader, loss_criterion_recon, loss_criterion_embed, ratio_loss, device=device)

            # Save the validation loss to a file
            checkpoint_name = os.path.basename(checkpoint_path)
            loss_filename = os.path.splitext(checkpoint_name)[0] + '_val_loss.pkl'
            loss_filepath = os.path.join(self.checkpoint_dir, loss_filename)
            val_dict = {'val_losses': avg_val_loss, 
                        'val_losses_embed': avg_val_loss_embed,
                          'val_losses_signal': avg_val_loss_signal}
            
            val_losses[checkpoint_path] = val_dict
            with open(loss_filepath, 'wb') as file:
                pickle.dump(val_dict, file)
        
        with open(os.path.join(self.checkpoint_dir, 'val_losses.pkl'), 'wb') as f:
            pickle.dump(val_losses, f)
    
    def validate(self, model, val_dataloader, loss_criterion_recon, loss_criterion_embed, ratio_loss, device='cuda'):
        model.eval()
        total_loss = 0.0
        total_loss_recon = 0.0
        total_loss_embed = 0.0
        total_samples = 0

        with torch.no_grad():
            for val_batch_id, val_data in enumerate(val_dataloader):
                val_eeg_data, val_psd = val_data
                val_eeg_data = val_eeg_data.to(device)
                val_psd = val_psd.to(device)

                conv_input, conv_output, s4_output, decoder_out, _ = model(val_eeg_data.clone())
                loss_recon = loss_criterion_recon(decoder_out, conv_input)
                loss_embed = loss_criterion_embed(s4_output, conv_output)
                loss = loss_recon + ratio_loss * loss_embed

                total_loss += loss.item() * val_eeg_data.size(0)
                total_loss_recon += loss_recon.item() * val_eeg_data.size(0)
                total_loss_embed += loss_embed.item() * val_eeg_data.size(0)
                logging.info(val_batch_id)
                total_samples += val_eeg_data.size(0)

        avg_val_loss = total_loss / total_samples
        avg_val_loss_recon = total_loss_recon / total_samples
        avg_val_loss_psd = total_loss_embed / total_samples
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss, avg_val_loss_recon, avg_val_loss_psd
    
    def train(self, model, train_dataloader, val_dataloader,
              optimizer, loss_criterion_recon, loss_criterion_embed, ratio_loss,
              resume: bool = False, resume_checkpoint: str = None,
              device: str = 'cuda'):

        model = model.to(device)
        model = self.resume_checkpoint(resume, resume_checkpoint, model)

        for epoch in range(self.start_epoch, self.n_epochs):
            logging.info(f"Epoch {epoch + 1}/{self.n_epochs}")

            model.train()
            total_loss = 0.0
            total_samples = 0

            for batch_id, data in enumerate(train_dataloader):
                eeg_data, psd = data
                psd = psd.to(device)

                self.batch_counter += 1

                eeg_data = eeg_data.to(device)
                optimizer.zero_grad()

                conv_input, conv_output, s4_output, decoder_out, _ = model(eeg_data.clone())
                loss_recon = loss_criterion_recon(decoder_out, eeg_data)
                loss_embed = loss_criterion_embed(s4_output, conv_output)
                loss = loss_recon + ratio_loss * loss_embed

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * eeg_data.size(0)
                total_samples += eeg_data.size(0)
                self.batch_loss += loss.item()
                self.batch_loss_recon = loss_recon.item()
                self.batch_loss_psd = loss_embed.item()

                logging.info(f'Epoch {epoch} Loss in batch {batch_id}: {loss.item()}')
                logging.info(f'Epoch {epoch} Recon Loss in batch {batch_id}: {loss_recon.item()}')
                logging.info(f'Epoch {epoch} Embed Loss in batch {batch_id}: {loss_embed.item()}')

                if self.batch_counter % 2000 == 0:
                    avg_batch_loss = self.batch_loss / 2000
                    avg_batch_loss_recon = self.batch_loss_recon / 2000
                    avg_batch_loss_psd = self.batch_loss_psd / 2000
                    
                    self.batch_loss_list.append(avg_batch_loss)
                    self.batch_loss_list_recon.append(avg_batch_loss_recon)
                    self.batch_loss_list_psd.append(avg_batch_loss_psd)

                    self.batch_loss = 0.0
                    self.batch_loss_recon = 0.0
                    self.batch_loss_psd = 0.0

                    # Save the batch loss list
                    batch_number = self.batch_counter
                    batch_losses_filename = f"{self.checkpoint_prefix}_batches_{batch_number - 1999}_{batch_number}_batch_losses.pkl"
                    batch_losses_path = os.path.join(self.checkpoint_dir, batch_losses_filename)
                    with open(batch_losses_path, 'wb') as file:
                        pickle.dump({'train_losses': self.batch_loss_list,
                                     'train_losses_recon': self.batch_loss_list_recon,
                                     'train_losses_psd': self.batch_loss_list_psd}, file)
                    
                    # Save the model checkpoint
                    model_checkpoint_filename = f"{self.checkpoint_prefix}_batches_{batch_number - 1999}_{batch_number}_avg_loss_{avg_batch_loss:.4f}.pt"
                    model_checkpoint_path = os.path.join(self.checkpoint_dir, model_checkpoint_filename)
                    
                    # Assuming you have a PyTorch model named 'model' that you want to save
                    torch.save(model.state_dict(), model_checkpoint_path)

            avg_loss = total_loss / total_samples
            logging.info(f"Train Loss: {avg_loss:.4f}")

            self.train_losses.append(avg_loss)

            avg_val_loss, _, _ = self.validate(model, val_dataloader, loss_criterion_recon, loss_criterion_embed, ratio_loss, device=device)
            self.val_losses.append(avg_val_loss)

            # Save the model and loss list every epoch
            model_filename = f"{self.checkpoint_prefix}_epoch_{epoch + 1}.pt"
            model_path = os.path.join(self.checkpoint_dir, model_filename)
            torch.save(model.state_dict(), model_path)

            losses_filename = f"{self.checkpoint_prefix}_epoch_{epoch + 1}_losses.pkl"
            losses_path = os.path.join(self.checkpoint_dir, losses_filename)
            with open(losses_path, 'wb') as file:
                pickle.dump({'train_losses': self.train_losses, 'val_losses': self.val_losses}, file)

        print("Training completed!")

if __name__ == '__main__':
    model, train_dataloader, val_dataloader, optimizer, loss_criterion = None, None, None, None, None
    # Usage
    # Initialize trainer instance
    trainer_instance = trainer(n_epochs=10, checkpoint_dir='checkpoints', checkpoint_prefix='model')
    # ... (create train and val dataloaders, model, optimizer, loss criterion, etc.)

    # Train the model
    trainer_instance.train(model, train_dataloader, val_dataloader, optimizer, loss_criterion, device='cuda')
