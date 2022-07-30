from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import timm

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
import pydicom
# from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
import utility


def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

def check_sample_img():
  img = utility.get_img('../data/train_images/1000015157.jpg')
  plt.imshow(img)
  plt.show()

  pass


def main():
  print(f"[trace] exec at main function")
  train = pd.read_csv('../data/train.csv')
  print(f'[trace] train.head: {train.head()}')
  print(f'[trace] train.label.value_counts(): {train.label.value_counts()}')
  #check_sample_img()
  seed_everything(utility.CFG['seed'])

  folds = StratifiedKFold(n_splits=utility.CFG['fold_num'], shuffle=True, random_state=utility.CFG['seed']).split(
    np.arange(train.shape[0]), train.label.values)

  for fold, (trn_idx, val_idx) in enumerate(folds):
    # we'll train fold 0 first
    if fold > 0:
      break

    print('Training with {} started'.format(fold))

    data_root = '../data/train_images/'
    print(len(trn_idx), len(val_idx))
    train_loader, val_loader = utility.prepare_dataloader(train, trn_idx, val_idx, data_root=data_root)

    device = torch.device(utility.CFG['device'])
    model = utility.CassvaImgClassifier(utility.CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=utility.CFG['lr'], weight_decay=utility.CFG['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=utility.CFG['T_0'], T_mult=1,
                                                                     eta_min=utility.CFG['min_lr'], last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25,
    #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

    loss_tr = nn.CrossEntropyLoss().to(device)  # MyCrossEntropyLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    opt_level = 'O0'
    from apex import amp
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    for epoch in range(utility.CFG['epochs']):
      utility.train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler,
                              schd_batch_update=False)

      with torch.no_grad():
        utility.valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)

      torch.save(model.state_dict(), '{}_fold_{}_{}'.format(utility.CFG['model_arch'], fold, epoch))

    # torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
    del model, optimizer, train_loader, val_loader, scaler, scheduler
    torch.cuda.empty_cache()

  pass


if __name__ == '__main__':
  main()
  pass
