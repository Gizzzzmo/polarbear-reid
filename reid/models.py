from typing import Sequence
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from itertools import chain
from .utils import has_func
from typing import List

"""
try:
    from detectron2 import modeling
    from detectron2.modeling.backbone import build_resnet_backbone
    from detectron2.config.config import CfgNode
    import pickle as pkl
    from collections import OrderedDict
    from detectron2.utils.file_io import PathManager
except:
    print("Warning: detectron2 library not found - some functionality will be unavailable")
"""

class MetricLearningNetwork(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        metric_embedder: nn.Module,
        classifiers: Sequence[nn.Module]
        ) -> None:
        super(MetricLearningNetwork, self).__init__()
        
        self.backbone = backbone
        self.metric_embedder = metric_embedder
        self.classifiers = nn.ModuleList(classifiers)
        
    def load_backbone(self, state_dict_location: str):
        if has_func(self.backbone, 'load'):
            self.backbone.load(state_dict_location)
        else:
            state_dict = torch.load(state_dict_location)
            self.backbone.load_state_dict(state_dict)
            
    def load_embedder(self, state_dict_location: str):
        if has_func(self.metric_embedder, 'load'):
            self.metric_embedder.load(state_dict_location)
        else:
            state_dict = torch.load(state_dict_location)
            self.metric_embedder.load_state_dict(state_dict)
            
    def load_classifier(self, state_dict_location:str, classifier_idx: int = 0):
        if has_func(self.classifiers[classifier_idx], 'load'):
            self.classifiers.load(state_dict_location)
        else:
            state_dict = torch.load(state_dict_location)
            self.classifiers[classifier_idx].load_state_dict(state_dict)
            
    def save_backbone(self, state_dict_location: str):
        if has_func(self.backbone, 'save'):
            self.backbone.save(state_dict_location)
        else:
            torch.save(self.backbone.state_dict(), state_dict_location)
    
    def save_metric_embedder(self, state_dict_location: str):
        if has_func(self.metric_embedder, 'save'):
            self.metric_embedder.save(state_dict_location)
        else:
            torch.save(self.metric_embedder.state_dict(), state_dict_location)
    
    def save_classifier(self, state_dict_location: str, classifier_idx: int = 0):
        if has_func(self.classifiers[classifier_idx], 'save'):
            self.classifiers.save(state_dict_location)
        else:
            torch.save(self.classifiers[classifier_idx].state_dict(), state_dict_location)
            
    def forward(self, x, classifier_idx=0, target='classifications'):
        backbone_output = None
        embedding_output = None
        classifier_output = None
        
        for _ in range(1):
            if 'embedding' not in x:
                if 'backbone' not in x:
                    backbone_output = self.backbone(x)
                else:
                    backbone_output = x['backbone']
                if target == 'backbone':
                    break
                
                embedding_output = self.metric_embedder(backbone_output)
            else:
                embedding_output = x['embedding']    
            if target == 'embedding':
                break
            
            classifier_output = self.classifiers[classifier_idx](embedding_output)
        
        return {
            'backbone': backbone_output,
            'embedding': embedding_output,
            'classifications': classifier_output
        }


class ResNetTrunk(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }
    
    def __init__(
        self,
        depth: int,
        pretrained: bool = True,
        instance_norm: bool = False
        ):
        super().__init__()
        
        if depth not in ResNetTrunk.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNetTrunk.__factory[depth](pretrained=pretrained)
        
        if instance_norm:
            for block in self.base.layer1:
                block.bn1 = torch.nn.InstanceNorm2d(
                    block.bn1.num_features,
                    block.bn1.eps,
                    block.bn1.momentum,
                    block.bn1.affine,
                    block.bn1.track_running_stats
                )
                block.bn2 = torch.nn.InstanceNorm2d(
                    block.bn2.num_features,
                    block.bn2.eps,
                    block.bn2.momentum,
                    block.bn2.affine,
                    block.bn2.track_running_stats
                )

            self.base.bn1 = torch.nn.InstanceNorm2d(
                self.base.bn1.num_features,
                self.base.bn1.eps,
                self.base.bn1.momentum,
                self.base.bn1.affine,
                self.base.bn1.track_running_stats
            )        
    
    def forward(self, x: dict, for_eval=False) -> torch.Tensor:
        x = x['imgs']
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        
        return x
    
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def ordered_parameters(self):
        module_groups = [
            chain(
                self.base.conv1.parameters(),
                self.base.bn1.parameters(),
                self.base.relu.parameters(),
                self.base.maxpool.parameters()
            )
        ]
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            for block in self.base._modules[layer]:
                module_groups.append(block.parameters())
                
        return module_groups
    
    def get_output_dimension(self):
        return self.base.fc.in_features

class SimpleMLP(nn.Module):
    def __init__(self, *layer_sizes, input_layer_size: int) -> None:
        super().__init__()
        
        layer_sizes = list(layer_sizes)
        layer_sizes.insert(0, input_layer_size)
        consecutive_sizes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.fcs = nn.Sequential(
            *(nn.Sequential(
                nn.Linear(inn, outt),
                nn.BatchNorm1d(outt),
                nn.ReLU()
            ) for inn, outt in consecutive_sizes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcs(x)
        

class MLPClassifier(nn.Module):
    def __init__(self, num_classes: int, depth: int = 1, num_features: int = 2048) -> None:
        
        super().__init__()
        
        self.fcs = SimpleMLP(*([num_features] * depth), input_layer_size=num_features)
        self.num_classes = num_classes
        self.classifier_x2 = nn.Linear(num_features, num_classes)
        nn.init.normal_(self.classifier_x2.weight, std=0.001)
        nn.init.constant_(self.classifier_x2.bias, 0)

        '''
        self.feat = nn.Linear(in_features, num_features, bias=False)
        self.feat_bn = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU(inplace=True)
        init.normal_(self.feat.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)
        self.drop = nn.Dropout(dropout)
        '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fcs(x)
        x = self.classifier_x2(x)
        return x

model_zoo = {
    "RESNET_TRUNK": ResNetTrunk,
    "MLP_CLASSIFIER": MLPClassifier,
    "SIMPLE_MLP": SimpleMLP    
}

def build_module_from_config(config: dict, *overwrite, **koverwrite) -> nn.Module:
    
    module_name = config['ARCHITECTURE']
    module_args = config.get('ARGS', []) + list(overwrite)
    module_kwargs = {**config.get('KWARGS', {}), **koverwrite}
    
    module = model_zoo[module_name](*module_args, **module_kwargs)
    return module

def build_metric_learning_network(config: dict, nums_classes: Sequence[int]) -> MetricLearningNetwork:
    backbone = build_module_from_config(config['BACKBONE'])
    output_dim = backbone.get_output_dimension()
    
    embedder = build_module_from_config(config['EMBEDDER'], input_layer_size=output_dim)
    classifiers = [build_module_from_config(config['CLASSIFIER'], num_classes=num_classes) for num_classes in nums_classes]
    
    model = MetricLearningNetwork(backbone, embedder, classifiers)
    return model