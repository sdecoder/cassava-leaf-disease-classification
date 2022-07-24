import argparse

import onnx
import tensorrt as trt
import os
import torch
import torchvision
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from enum import Enum

from tqdm import tqdm

import utility

def allocate_buffers_for_encoder(engine):

  print('[trace] reach func@allocate_buffers')
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()

  binding_to_type = {}
  binding_to_type['input'] = np.float32
  binding_to_type['output'] = np.float32

  for binding in engine:
    print(f'[trace] current binding: {str(binding)}')
    _binding_shape = engine.get_binding_shape(binding)
    _volume = trt.volume(_binding_shape)
    size = _volume #* engine.max_batch_size
    print(f'[trace] current binding size: {size}')
    dtype = binding_to_type[str(binding)]
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(utility.HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(utility.HostDeviceMem(host_mem, device_mem))
  return inputs, outputs, bindings, stream


def test_using_trt(engine, test_loader):

  print(f'[trace] run the test using TensorRT engine')
  inputs, outputs, bindings, stream = allocate_buffers_for_encoder(engine)
  context = engine.create_execution_context()
  batch_size = 16

  loss_sum = 0
  sample_num = 0
  image_preds_all = []
  image_targets_all = []
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  loss_fn = nn.CrossEntropyLoss().to(device)

  pbar = tqdm(enumerate(test_loader), total=len(test_loader))
  batch_num = 20
  for step, (imgs, image_labels) in pbar:
    if step >= batch_num:
      break
    imgs = imgs.to(device).float()
    image_labels = image_labels.long()

    np.copyto(inputs[0].host, utility.to_numpy(imgs).ravel())
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    image_preds = outputs[0].host.reshape((16, 5))
    image_preds = torch.from_numpy(image_preds)
    image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
    image_targets_all += [image_labels.detach().cpu().numpy()]

    loss = loss_fn(image_preds, image_labels)
    loss_sum += loss.item() * image_labels.shape[0]
    sample_num += image_labels.shape[0]
    description = f'step {step} loss: {loss_sum / sample_num:.4f}'
    pbar.set_description(description)

  image_preds_all = np.concatenate(image_preds_all)
  image_targets_all = np.concatenate(image_targets_all)
  print('[trace] validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

  pass

def _prepare_test_loader():

  data_root = '../data/train_images/'
  import pandas as pd
  train = pd.read_csv('../data/train.csv')
  train_ds = utility.CassavaDataset(train, data_root,
                                    transforms=utility.get_train_transforms(),
                                    output_label=True,
                                    one_hot_label=False,
                                    do_fmix=False,
                                    do_cutmix=False)


  train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=utility.CFG['train_bs'],
    pin_memory=False,
    drop_last=False,
    shuffle=True,
    num_workers=utility.CFG['num_workers'],
    # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
  )
  return train_loader

def _validate_trt_engine_file(filename):

  print(f'[trace] validate the tensorrt engine file using {filename}')
  if not os.path.exists(filename):
    print(f'[trace] target engine file {filename} does not exist, exit')
    return

  print(f'[trace] start to read the engine data')
  with open(filename, 'rb') as f:
    engine_data = f.read()
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  trt_runtime = trt.Runtime(TRT_LOGGER)
  engine = trt_runtime.deserialize_cuda_engine(engine_data)
  if engine:
    print(f'[trace] TensorRT engine created')
  else:
    print(f'[trace] failed to create TensorRT engine, exit')
    exit(-1)

  _test_loader = _prepare_test_loader()
  test_using_trt(engine, _test_loader)
  pass

def _main():
  print(f"[trace] start@main")
  engine_modes = []
  engine_modes.append(utility.CalibratorMode.INT8)
  engine_modes.append(utility.CalibratorMode.FP16)
  engine_modes.append(utility.CalibratorMode.TF32)
  engine_modes.append(utility.CalibratorMode.FP32)

  for _mode in engine_modes:
    realname = f'../models/efficientnet_b4_ns.{_mode.name}.engine'
    _validate_trt_engine_file(realname)


  print(f"[trace] end@main")
  pass

if __name__ == '__main__':
  _main()
  pass
