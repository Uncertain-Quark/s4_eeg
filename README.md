# Knowledge-guided EEG Representation Learning
Code implementation for [EEG SSL Paper](https://arxiv.org/abs/2403.03222) accepted at EMBC 2024.

## Usage
- ```main_no_norm_patch.py```: Pre-train models using the TUEG data using reconstruction loss only.
- ```main_no_norm_patch_multitask.py```: Pre-train models using combination of knowledge guided loss and reconstruction loss.

## Pretrained Model Checkpoints
Find the pre-trained model checkpoints at this [link](https://drive.google.com/drive/folders/1k7tcnVaZELQMGu1uYtcDgJBwFQNLkUtn?usp=drive_link)

## Datasets
### Pretrained Dataset
[TUEG Dataset](https://isip.piconepress.com/projects/tuh_eeg/) has been used for pre-training. The pre-processing pipeline is as mentioned in the paper.
### Fine-tuning Datasets
- [MMI Physionet](https://www.physionet.org/content/eegmmidb/1.0.0/)
- [BCI IV 2A](https://www.bbci.de/competition/iv/)
## Acknowledgements
Parts of the code (S4 implementation) are taken from [link](https://github.com/state-spaces/s4)
