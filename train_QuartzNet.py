# -*- coding: utf-8 -*-

import os
import glob
import subprocess
import tarfile
import wget
import copy
from omegaconf import OmegaConf, open_dict

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager
from collections import defaultdict
import torch
import torch.nn as nn
import pytorch_lightning as ptl


def get_charset():
    charset = defaultdict(int)
    text = ' abcdefghijklmnopqrstuvwxyzàèéìòù'	# ITALIAN!
    for character in text:
        charset[character] += 1
        if (character == 'z'):
            charset["'"] += 1
    return charset

charset = list(get_charset().keys())
print('CHARSET',charset)


# FINE TUNING STT_EN
char_model = nemo_asr.models.ASRModel.restore_from(restore_path="models/stt_en_quartznet15x5.nemo", map_location='cpu')

"""## Update the vocabulary
"""

char_model.change_vocabulary(new_vocabulary=list(charset))

#@title Freeze Encoder { display-mode: "form" }
freeze_encoder = False #True #@param ["False", "True"] {type:"raw"}
freeze_encoder = bool(freeze_encoder)

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

if freeze_encoder:
  char_model.encoder.freeze()
  char_model.encoder.apply(enable_bn_se)
  logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")
else:
  char_model.encoder.unfreeze()
  logging.info("Model encoder has been un-frozen")

"""## Update config
"""

with open_dict(char_model.cfg):
  char_model.cfg.labels = list(charset)
  char_model.cfg.sample_rate = 16000

cfg = copy.deepcopy(char_model.cfg)

"""### Setting up data loaders
"""

train_manifest_cleaned = './TCorpora/CV7_MLS_V_A.json'
dev_manifest_cleaned   = './TCorpora/cv-corpus-7.0-2021-07-21_dev.json'

# Setup train, validation, test configs
with open_dict(cfg):    
  # Train dataset  
  cfg.train_ds.manifest_filepath = train_manifest_cleaned
  cfg.train_ds.labels = list(charset)
  cfg.train_ds.normalize_transcripts = False
  cfg.train_ds.batch_size = 96 #IT DEPENDS ON GPU MEMORY (96->32GB)
  cfg.train_ds.num_workers = 8
  cfg.train_ds.pin_memory = True
  cfg.train_ds.trim_silence = True

  # Validation dataset
  cfg.validation_ds.manifest_filepath = dev_manifest_cleaned
  cfg.validation_ds.labels = list(charset)
  cfg.validation_ds.normalize_transcripts = False
  cfg.validation_ds.batch_size = 4
  cfg.validation_ds.num_workers = 4
  cfg.validation_ds.pin_memory = True
  cfg.validation_ds.trim_silence = True

# setup data loaders with new configs
char_model.setup_training_data(cfg.train_ds)
char_model.setup_multiple_validation_data(cfg.validation_ds)

"""### Setting up optimizer and scheduler
"""

with open_dict(char_model.cfg.optim):
  #char_model.cfg.optim.name = novograd
  char_model.cfg.optim.lr = 0.0012
  char_model.cfg.optim.betas = [0.8,0.5] 
  char_model.cfg.optim.weight_decay = 0.001  
  #char_model.cfg.optim.sched.name = CosineAnnealing
  char_model.cfg.optim.sched.warmup_steps = 500  
  #char_model.cfg.optim.sched.warmup_ratio = 0.05  # 5 % warmup
  char_model.cfg.optim.sched.min_lr = 1e-6
print('OPTIM:',OmegaConf.to_yaml(char_model.cfg.optim))

"""### Setting up augmentation
"""

char_model.spec_augmentation = char_model.from_config_dict(char_model.cfg.spec_augment)

"""## Setup Metrics
"""

char_model._wer.use_cer = False
char_model._wer.log_prediction = True

"""## Setup Trainer and Experiment Manager
"""

if torch.cuda.is_available():
  gpus = 1
else:
  gpus = 0

EPOCHS = 256 #512

trainer = ptl.Trainer(gpus=gpus, 
                      max_epochs=EPOCHS, 
                      accumulate_grad_batches=1,
                      checkpoint_callback=False,
                      logger=False,
                      log_every_n_steps=100,
                      check_val_every_n_epoch=1,
                      amp_level='O1', 
                      precision=16)   


# Setup model with the trainer
char_model.set_trainer(trainer)

# Finally, update the model's internal config
char_model.cfg = char_model._cfg
print('-----------------------------------------------------------')
print('FINAL CONFIG:')
print(OmegaConf.to_yaml(char_model.cfg))
print('-----------------------------------------------------------')

# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
os.environ.pop('NEMO_EXPM_VERSION', None)

LANGUAGE = 'italian'
config = exp_manager.ExpManagerConfig(
    exp_dir=f'experiments/lang-{LANGUAGE}/',
    name=f"ASR-Char-Model-Language-{LANGUAGE}",
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

config = OmegaConf.structured(config)

logdir = exp_manager.exp_manager(trainer, config)

trainer.fit(char_model)
