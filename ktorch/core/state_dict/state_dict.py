from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, List, Any

import torch
import torch.nn as nn


__all__ = ['StateDictTransform', 'Identity', 'RemovePrefix', 
           'AddPrefix', 'IntersectWith', 'IncompatibleKeys',
           'load_state_dict', 'check_state_dict_keys']


class StateDictTransform(ABC):
    @abstractmethod
    def __call__(self, old_value: Mapping[str, Any]) -> Mapping[str, Any]:
        pass
    
    
class Identity(StateDictTransform):
    def __call__(self, old_value: Mapping[str, Any]) -> Mapping[str, Any]:
        return old_value
    
    
class RemovePrefix(StateDictTransform):
    def __init__(self, prefix):
        self.prefix = prefix
    
    def __call__(self, state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(self.prefix):
                k = k[len(self.prefix):]
            new_state_dict[k] = v
        return new_state_dict
        

class AddPrefix(StateDictTransform):
    def __init__(self, prefix):
        self.prefix = prefix
    
    def __call__(self, state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[self.prefix + k] = v
        return new_state_dict
        

class IntersectWith(StateDictTransform):
    def __init__(self, other: Mapping[str, Any]):
        self.other = other
        
    def __call__(self, state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in self.other and v.shape == self.other[k].shape:
                new_state_dict[k] = v
        return new_state_dict

        
@dataclass
class IncompatibleKeys:
    missing_keys: List[str]
    unexpected_keys: List[str]
    
    
def load_state_dict(model: nn.Module, state_dict: Mapping[str, Any], strict: bool = True, 
                    state_dict_trans: StateDictTransform = None) -> IncompatibleKeys:
    state_dict_trans = state_dict_trans or Identity()
    new_state_dict = state_dict_trans(state_dict)
    incompatible_keys = model.load_state_dict(new_state_dict, strict=strict)
    return IncompatibleKeys(incompatible_keys.missing_keys, incompatible_keys.unexpected_keys)


def check_state_dict_keys(model, ckpt_state_dict):
    assert isinstance(model, (torch.nn.Module, dict))
    assert isinstance(ckpt_state_dict, dict)

    ckpt_keys = set(ckpt_state_dict.keys())
    if isinstance(model, torch.nn.Module):
        model_keys = set(model.state_dict().keys())
    else:
        model_keys = set(model.keys())
    used_ckpt_keys = model_keys & ckpt_keys
    unused_ckpt_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Used keys:              {}'.format(len(used_ckpt_keys)))
    print('Missing keys:           {}'.format(len(missing_keys)))
    print('Unused checkpoint keys: {}'.format(len(unused_ckpt_keys)))
    assert len(used_ckpt_keys) > 0, 'load NONE from pretrained checkpoint'

