# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.nets.network import Network
from lib.model.config import cfg

import lib.utils.timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck

class ResNet(torchvision.models.resnet.ResNet):
  def __init__(self, block, layers, num_classes=1000, image_channels=3):
    self.inplanes = 64
    super(ResNet, self).__init__(block, layers, num_classes)

    self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)

    # change to match the caffe resnet
    for i in range(2, 4):
      getattr(self, 'layer%d'%i)[0].conv1.stride = (2,2)
      getattr(self, 'layer%d'%i)[0].conv2.stride = (1,1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    self.layer4[0].conv2.stride = (1,1)
    self.layer4[0].downsample[0].stride = (1,1)

    del self.avgpool, self.fc


def resnet18(pretrained=False, image_channels=3):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], image_channels=image_channels)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False, image_channels=3):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], image_channels=image_channels)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False, image_channels=3):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], image_channels=image_channels)
  print("pretrained? ", "yes" if pretrained else "no")
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False, image_channels=3):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], image_channels=image_channels)
  print("pretrained? ", "yes" if pretrained else "no")
  if pretrained:
    if image_channels == 3:
      model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    elif image_channels == 4:
      model.load_state_dict_rgba(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False, image_channels=3):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3], image_channels=image_channels)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnetv1(Network):
  def __init__(self, num_layers=50, image_channels=3):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._num_layers = num_layers
    self._net_conv_channels = 1024
    self._fc7_channels = 2048
    self._image_channels = image_channels

  def _crop_pool_layer(self, bottom, rois):
    return Network._crop_pool_layer(self, bottom, rois, cfg.RESNET.MAX_POOL)

  def _image_to_head(self):
    net_conv = self._layers['head'](self._image)
    self._act_summaries['conv'] = net_conv

    return net_conv

  def _head_to_tail(self, pool5):
    fc7 = self.resnet.layer4(pool5).mean(3).mean(2) # average pooling after layer4
    return fc7

  def _init_head_tail(self):
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      self.resnet = resnet50(image_channels=self._image_channels)

    elif self._num_layers == 101:
      self.resnet = resnet101(image_channels=self._image_channels)

    elif self._num_layers == 152:
      self.resnet = resnet152(image_channels=self._image_channels)

    else:
      # other numbers are not supported
      raise NotImplementedError

    # Fix blocks 
    for p in self.resnet.bn1.parameters(): p.requires_grad=False
    for p in self.resnet.conv1.parameters(): p.requires_grad=False
    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.resnet.layer3.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.resnet.layer2.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.resnet.layer1.parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.resnet.apply(set_bn_fix)

    # Build resnet.
    self._layers['head'] = nn.Sequential(self.resnet.conv1, self.resnet.bn1,self.resnet.relu, 
      self.resnet.maxpool,self.resnet.layer1,self.resnet.layer2,self.resnet.layer3)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode (not really doing anything)
      self.resnet.eval()
      if cfg.RESNET.FIXED_BLOCKS <= 3:
        self.resnet.layer4.train()
      if cfg.RESNET.FIXED_BLOCKS <= 2:
        self.resnet.layer3.train()
      if cfg.RESNET.FIXED_BLOCKS <= 1:
        self.resnet.layer2.train()
      if cfg.RESNET.FIXED_BLOCKS == 0:
        self.resnet.layer1.train()

      # Set batchnorm always in eval mode during training
      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.resnet.apply(set_bn_eval)

  def load_pretrained_cnn_rgba(self, state_dict):
    model_dict = self.resnet.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in state_dict.items() if (k in model_dict and not (k == 'conv1.weight'))}
    # for k, v in state_dict.items():
      # print(k)
      # if k == 'conv1.weight':
      #   print("here!")
      #   print(v.shape)
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    self.resnet.load_state_dict(model_dict)


  def load_pretrained_cnn(self, state_dict):
    self.resnet.load_state_dict({k: state_dict[k] for k in list(self.resnet.state_dict())}, strict=False)

