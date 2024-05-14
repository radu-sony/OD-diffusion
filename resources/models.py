# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.backbones.csp_darknet import CSPLayer, Focus
from mmdet.utils import ConfigType, OptMultiConfig
from mmdet.models.data_preprocessors import DetDataPreprocessor

from mmyolo.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv, SPPFBottleneck
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.backbones.csp_darknet import YOLOv8CSPDarknet
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

import numpy as np

def rggb_to_rgb(data):
    #assert data.shape[1] == 4
    if len(data.shape) == 4:
        img_r = data[:, 0]
        img_g1 = data[:, 1]
        img_g2 = data[:, 2]
        img_b = data[:, 3]
    else:
        img_r = data[0]
        img_g1 = data[1]
        img_g2 = data[2]
        img_b = data[3]        

    img_g = (img_g1 + img_g2) / 2

    img = torch.stack((img_r, img_g, img_b), axis=1).float()
    return img

@MODELS.register_module()
class YOLOv5DetDataPreprocessor_RAWtoRGB(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolov5_collate`
    """

    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            inputs, data_samples = data['inputs'], data['data_samples']
            inputs = torch.stack(inputs).cuda()
            inputs = torch.clip((inputs.float() - self.mean) / self.std, 0, 1)
            inputs = rggb_to_rgb(inputs)
            data = {'inputs': inputs, 'data_samples': data_samples}
            return data

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        
        #assert isinstance(data['data_samples'], dict)

        # TODO: Supports multi-scale training
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = torch.clip((inputs.float() - self.mean) / self.std, 0, 1)

        inputs = rggb_to_rgb(inputs)

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)


        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'img_metas': img_metas
        }
        if 'masks' in data_samples:
            data_samples_output['masks'] = data_samples['masks']

        return {'inputs': inputs, 'data_samples': data_samples_output}

        # inputs, data_samples = data['inputs'], data['data_samples']
        # inputs = rggb_to_rgb(inputs)
        # data = {'inputs': inputs.cuda(), 'data_samples': data_samples}
        # return data
        


@MODELS.register_module()
class YOLOv8CSPDarknetPreproc(YOLOv8CSPDarknet):

    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 preproc: str = '',
                 preproc_params: List = [],
                 init_cfg: OptMultiConfig = None):

        self.arch_settings[arch][-1][1] = last_stage_out_channels
        self.preproc = preproc

        super().__init__(
            arch=arch,
            last_stage_out_channels=last_stage_out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            preproc=preproc,
            preproc_params=preproc_params,
            init_cfg=init_cfg)
    
    def build_preproc_layer(self, preproc_params=[]) -> nn.Module:
        print('!!!!!!!!!!!!!!!!!!!!!!!! Building preproc layer', preproc_params)
        if preproc_params['name'] == 'preproc':
            preproc_layer = preproc_model(preproc_params=preproc_params).cuda()
        elif preproc_params['name'] == 'gamma-cnn':
            preproc_layer = gamma_cnn(preproc_params=preproc_params).cuda()
        elif preproc_params['name'] == 'norm':
            preproc_layer = preproc_model_norm(preproc_params=preproc_params).cuda()
        elif preproc_params['name'] == 'norm-max':
            preproc_layer = preproc_model_norm_max(preproc_params=preproc_params).cuda()
        elif preproc_params['name'] == 'norm-max-cnn':
            preproc_layer = preproc_model_norm_max_cnn(preproc_params=preproc_params).cuda()
        elif preproc_params['name'] == 'norm-gamma':
            preproc_layer = preproc_model_norm_gamma(preproc_params=preproc_params).cuda()
        elif preproc_params['name'] == 'norm-fundiv':
            preproc_layer = preproc_model_norm_fundiv(preproc_params=preproc_params).cuda()
        elif preproc_params['name'] == 'dynamic-gamma':
            preproc_layer = preproc_model_dynamic_gamma(preproc_params=preproc_params).cuda()
        elif preproc_params['name'] == 'dynamic-fundiv':
            preproc_layer = preproc_model_dynamic_fundiv(preproc_params=preproc_params).cuda()
        elif preproc_params['name'] == 'load':
            preproc_layer = preproc_model().cuda()
            link = '/home/radu/work/ISP/ISP2/outputs/ISP_e64_lr0.001_b64_1688976820_1/ISP_e64_lr0.001_b64_1688976820_1.pth'
            preproc_layer.load_state_dict(torch.load(link))
        return preproc_layer

class gamma_cnn(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()

        print('-> Building simple gamma max layer.')

        self.gamma = preproc_params['gamma']
        self.trainable = preproc_params['trainable']

        self.hidden_channels = preproc_params['hidden_channels']
        self.input_channels = preproc_params['input_channels']

        self.rggb_max = torch.tensor(preproc_params['rggb_max'], requires_grad=False).cuda()

        self.cnn1 = nn.Sequential(nn.Conv2d(self.input_channels, 
                                            self.input_channels*self.hidden_channels, 
                                            kernel_size=1, 
                                            stride=1, 
                                            bias=True, 
                                            groups=self.input_channels),
                                  nn.ReLU())
        self.cnn2 = nn.Sequential(nn.Conv2d(self.input_channels*self.hidden_channels,
                                            self.input_channels,  
                                            kernel_size=1, 
                                            stride=1, 
                                            bias=False, 
                                            groups=self.input_channels),
                                  nn.ReLU())

        self.init_params()
    
    def init_params(self):
        weights1 = np.ones((self.input_channels*self.hidden_channels,1,1,1), dtype=np.float32)
        weights2, biases = self.make_cnn_gamma()
        self.cnn1[0].weight = nn.Parameter(torch.from_numpy(weights1).float().cuda())
        self.cnn1[0].bias = nn.Parameter(torch.from_numpy(biases).cuda())
        self.cnn2[0].weight = nn.Parameter(torch.from_numpy(weights2).cuda())

        if not self.trainable:
            self.cnn1[0].weight.requires_grad = False
            self.cnn1[0].bias.requires_grad = False
            self.cnn2[0].weight.requires_grad = False    
        else:
            self.cnn1[0].weight.requires_grad = False
            self.cnn1[0].bias.requires_grad = True
            self.cnn2[0].weight.requires_grad = True                
        
    def fun(self, x, gamma):
        return x**gamma

    def make_cnn_gamma(self):
        bias = np.linspace(0, 1, self.hidden_channels+1)
        bias = bias ** 2 # since gamma is 'bendy' close to 0, have more bias values close to 0.

        weights = np.zeros(self.hidden_channels)

        for i in range(len(weights)):
            w = (self.fun(bias[i+1], self.gamma)-self.fun(bias[i], self.gamma)) / (bias[i+1]-bias[i]) - np.sum(weights)
            weights[i] = w

        weights = np.tile(weights, (self.input_channels, 1))
        weights = np.expand_dims(np.expand_dims(weights, axis=-1), axis=-1).astype(np.float32)
        biases = np.concatenate([bias[:-1]]*self.input_channels).astype(np.float32) * -1

        return weights, biases

    def forward(self, x):
        x = x / self.rggb_max
        x = self.cnn1(x)
        x = self.cnn2(x)
        return x   


class preproc_model(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()

        self.gamma = preproc_params['gamma']
        self.mean = preproc_params['mean']
        self.std = preproc_params['std']

    def forward(self, x):
        x = x ** self.gamma
        return x


class preproc_model_norm(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()
        print('-> Building simple norm layer.')
        self.rggb_max = torch.tensor(preproc_params['rggb_max'], requires_grad=False).cuda()
        self.mean = torch.tensor(preproc_params['mean']).view(1,len(preproc_params['mean']),1,1).cuda()
        self.std = torch.tensor(preproc_params['std']).view(1,len(preproc_params['std']),1,1).cuda()

    def forward(self, x):
        x = x / self.rggb_max
        x = (x - self.mean) / self.std
        return x

class preproc_model_norm_max(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()
        print('-> Building simple norm max layer.')
        self.rggb_max = torch.tensor(preproc_params['rggb_max'], requires_grad=False).cuda()

    def forward(self, x):
        x = x / self.rggb_max
        #print(x[0,:,320,320])
        return x

class preproc_model_norm_max_cnn(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()
        print('-> Building simple norm max layer.')
        self.rggb_max = torch.tensor(preproc_params['rggb_max'], requires_grad=False).cuda()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=1, stride=1, bias=False),
                                 nn.LeakyReLU(0.1),
                                 nn.Conv2d(32, 4, kernel_size=1, stride=1, bias=False),
                                 nn.LeakyReLU(0.1))

        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        x = x / self.rggb_max
        x = self.cnn(x)
        return x

class preproc_model_norm_gamma(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()
        self.rggb_max = torch.tensor(preproc_params['rggb_max'], requires_grad=False).cuda()
        self.mean = torch.tensor(preproc_params['mean'], requires_grad=False).view(1,len(preproc_params['mean']),1,1).cuda()
        self.std = torch.tensor(preproc_params['std'], requires_grad=False).view(1,len(preproc_params['std']),1,1).cuda()
        self.gamma = torch.tensor(preproc_params['value'], requires_grad=False)

    def forward(self, x):
        x = x / self.rggb_max
        x = torch.pow(x, self.gamma)
        x = (x - self.mean) / self.std
        return x

class preproc_model_norm_fundiv(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()
        self.rggb_max = torch.tensor(preproc_params['rggb_max'], requires_grad=False).cuda()
        self.mean = torch.tensor(preproc_params['mean'], requires_grad=False).view(1,len(preproc_params['mean']),1,1).cuda()
        self.std = torch.tensor(preproc_params['std'], requires_grad=False).view(1,len(preproc_params['std']),1,1).cuda()
        self.alpha = torch.tensor(preproc_params['value'], requires_grad=False)

    def forward(self, x):
        x = x / self.rggb_max
        x = x * (1 + self.alpha)/(x + self.alpha)
        x = (x - self.mean) / self.std
        return x

class preproc_model_dynamic_fundiv(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()
        self.rggb_max = torch.tensor(preproc_params['rggb_max'], requires_grad=False).cuda()

        self.params = preproc_feature_extractor(preproc_params=preproc_params).cuda()

    def forward(self, x):
        x = x / self.rggb_max
        params = self.params(x)
        params = params.unsqueeze(-1).unsqueeze(-1)
        x = x * (1 + params[:,0:1])/(x + params[:,0:1])
        x = (x - params[:,1:2]) / params[:,2:]
        return x

class preproc_model_dynamic_gamma(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()
        self.rggb_max = torch.tensor(preproc_params['rggb_max'], requires_grad=False).cuda()

        self.params = preproc_feature_extractor(preproc_params=preproc_params).cuda()

    def forward(self, x):
        x = x / self.rggb_max
        params = self.params(x)
        params = params.unsqueeze(-1).unsqueeze(-1)
        x = torch.clip(x, 1e-6, 1)
        x = torch.pow(x, params[:,0:1])
        x = (x - params[:,1:2]) / params[:,2:]
        return x

class preproc_feature_extractor(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()

        self.max_reduce = nn.MaxPool2d(kernel_size=(4,4), stride=4)
        self.avg_reduce = nn.AvgPool2d(kernel_size=(4,4), stride=4)
        bias = False
        self.mlp_layers = []
        self.mlp_layers.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                             nn.Flatten(start_dim=1, end_dim=-1),
                                             nn.Linear(8, preproc_params['hidden_size'], bias=bias),
                                             nn.LeakyReLU(0.1)))
        for i in range(preproc_params['n_layers']-2):
            self.mlp_layers.append(nn.Sequential(nn.Linear(preproc_params['hidden_size'], preproc_params['hidden_size'], bias=bias),
                                                 nn.LeakyReLU(0.1)))
        self.mlp_layers.append(nn.Sequential(nn.Linear(preproc_params['hidden_size'], preproc_params['n_params'], bias=bias),
                                                 nn.Sigmoid()))
        self.mlp_layers = nn.ModuleList(self.mlp_layers)
        
        for seq in self.mlp_layers:
            for layer in seq.children():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)
    
    def forward(self, x):
        m = self.max_reduce(x)
        a = self.avg_reduce(x)
        x = torch.cat([m,a], dim=1)
        for layer in self.mlp_layers:
            x = layer(x)
        return x