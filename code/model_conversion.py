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

import utility


def export2onnx(model):
  print(f'[trace] working in the func: create_onnx_file')
  onnx_file_name = '../models/efficientnet_b4_ns.onnx'
  if os.path.exists(onnx_file_name):
    print(f'[trace] {onnx_file_name} exist, return')
    return onnx_file_name

  print(f'[trace] start to export the torchvision resnet50')
  input_name = ['input']
  output_name = ['output']
  from torch.autograd import Variable

  '''
  import torch.nn as nn
  # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  model.fc = nn.Linear(2048, 10, bias=True)
  '''

  # Input to the model
  # [trace] input shape: torch.Size([16, 3, 512, 512])
  batch_size = 16
  channel = 3
  pic_dim = 512

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input = torch.randn(batch_size, channel, pic_dim, pic_dim, requires_grad=True)
  input = input.to(device)
  '''
  output = model(input)
  print(f'[trace] model output: {output.size()}')
  '''

  # Export the model
  torch.onnx.export(model,  # model being run
                    input,  # model input (or a tuple for multiple inputs)
                    onnx_file_name,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=16,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model's input names
                    output_names=['output'],  # the model's output names
                    dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                  'output': {0: 'batch_size'}})

  print(f'[trace] done with onnx file exporting')
  # modify the network to adapat MNIST
  pass


TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def GiB(val):
  return val * 1 << 30


def build_engine_common_routine(network, builder, config, runtime, engine_file_path):
  input_batch_size = 16
  input_channel = 3
  input_image_width = 512
  input_image_height = input_image_width
  network.get_input(0).shape = [input_batch_size, input_channel, input_image_width, input_image_height]
  plan = builder.build_serialized_network(network, config)
  if plan == None:
    print("[trace] builder.build_serialized_network failed, exit -1")
    exit(-1)
  engine = runtime.deserialize_cuda_engine(plan)
  print("[trace] Completed creating Engine")
  with open(engine_file_path, "wb") as f:
    f.write(plan)
  return engine


class Calibrator(trt.IInt8EntropyCalibrator2):

  def __init__(self, training_dataset, cache_file, batch_size=16):
    # Whenever you specify a custom constructor for a TensorRT class,
    # you MUST call the constructor of the parent explicitly.
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.cache_file = cache_file
    # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.

    self.data_provider = training_dataset
    self.batch_size = batch_size
    self.current_index = 0

    # Allocate enough memory for a whole batch.
    channel = 3
    dimension = 512
    single_data_bytes = channel * dimension * dimension
    self.device_input = cuda.mem_alloc(single_data_bytes * self.batch_size * 4)

  def get_batch_size(self):
    return self.batch_size

  # TensorRT passes along the names of the engine bindings to the get_batch function.
  # You don't necessarily have to use them, but they can be useful to understand the order of
  # the inputs. The bindings list is expected to have the same ordering as 'names'.
  def get_batch(self, names):

    print(f'[trace] the names.type? {type(names)}, value: {names}')
    max_data_item = len(self.data_provider)
    if self.current_index + self.batch_size > max_data_item:
      return None

    current_batch = int(self.current_index / self.batch_size)
    if current_batch % 10 == 0:
      print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

    imgs, labels = next(iter(self.data_provider))
    #batch = self.data[self.current_index: self.current_index + self.batch_size].ravel()
    _elements = imgs.ravel().numpy()
    cuda.memcpy_htod(self.device_input, _elements)
    self.current_index += self.batch_size
    return [self.device_input]

  def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
    print(f'[trace] MNISTEntropyCalibrator: read_calibration_cache: {self.cache_file}')
    if os.path.exists(self.cache_file):
      with open(self.cache_file, "rb") as f:
        return f.read()

  def write_calibration_cache(self, cache):
    print(f'[trace] MNISTEntropyCalibrator: write_calibration_cache: {cache}')
    with open(self.cache_file, "wb") as f:
      f.write(cache)



def generate_trt_engine():

  calibration_cache = "calibration.cache"
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

  calib = Calibrator(train_loader, cache_file=calibration_cache)
  # engine = build_engine_from_onnxmodel_int8(onnxmodel, calib)
  mode: utility.CalibratorMode = utility.CalibratorMode.FP16
  onnx_file_path = '../models/efficientnet_b4_ns-sim.onnx'

  if not os.path.exists(onnx_file_path):
    print(f'[trace] target file {onnx_file_path} not exist, exiting')
    exit(-1)

  batch_size = 16
  with trt.Builder(TRT_LOGGER) as builder, \
      builder.create_network(EXPLICIT_BATCH) as network, \
      builder.create_builder_config() as config, \
      trt.OnnxParser(network, TRT_LOGGER) as parser, \
      trt.Runtime(TRT_LOGGER) as runtime:

    config.max_workspace_size = 1 << 28  # 256MiB
    builder.max_batch_size = batch_size
    # Parse model file
    print("[trace] loading ONNX file from path {}...".format(onnx_file_path))
    with open(onnx_file_path, "rb") as model:
      print("[trace] beginning ONNX file parsing")
      if not parser.parse(model.read()):
        print("[error] failed to parse the ONNX file.")
        for error in range(parser.num_errors):
          print(parser.get_error(error))
        return None
      print("[trace] completed parsing of ONNX file")

    builder.max_batch_size = batch_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(4))

    if mode == utility.CalibratorMode.INT8:
      config.set_flag(trt.BuilderFlag.INT8)
    elif mode == utility.CalibratorMode.FP16:
      config.set_flag(trt.BuilderFlag.FP16)
    elif mode == utility.CalibratorMode.TF32:
      config.set_flag(trt.BuilderFlag.TF32)
    elif mode == utility.CalibratorMode.FP32:
      # do nothing since this is the default branch
      # config.set_flag(trt.BuilderFlag.FP32)
      pass
    else:
      print(f'[trace] unknown calibrator mode: {mode.name}, exit')
      exit(-1)

    config.int8_calibrator = calib
    engine_file_path = f'../models/efficientnet_b4_ns.{mode.name}.engine'
    return build_engine_common_routine(network, builder, config, runtime, engine_file_path)
  pass


def load_trained_model():
  print(f'[trace] exec@load_trained_model')
  import pandas as pd
  train = pd.read_csv('../data/train.csv')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = utility.CassvaImgClassifier(utility.CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)

  pt_file_path = '../models/tf_efficientnet_b4_ns_fold_0_5'
  model.load_state_dict(torch.load(pt_file_path))
  model.eval()
  print(f'[trace] model info: \n{model}')
  print(f'[trace] end@load_trained_model')
  return model


def main():
  model = load_trained_model()
  export2onnx(model)
  generate_trt_engine()
  pass


if __name__ == '__main__':
  main()
  pass
