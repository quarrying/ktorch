import os
import glob
import socket
import random
import datetime
import functools
from dataclasses import dataclass
from typing import List, Union, Optional

import torch
import khandy
import numpy as np


__all__ = ['sum_tensor_list', 'get_latest_model_path', 'get_run_name', 
           'get_run_save_dir', 'set_random_seed', 'ModelParameterInfo',
           'convert_parameter_infos_to_markdown_table', 'get_model_parameter_infos']


def sum_tensor_list(tensor_list):
    """
    References:
        tf.math.add_n
    """
    return functools.reduce(lambda acc, x: acc.add_(x), tensor_list,
                            torch.zeros_like(tensor_list[0]))
                            
                            
def get_latest_model_path(model_path, extension='pth'):
    """
    References:
        tf.train.latest_checkpoint
    """
    extension = khandy.normalize_extension(extension)
    if model_path is not None:
        model_path = os.path.expanduser(model_path)
        if os.path.isdir(model_path):
            filenames = glob.glob(os.path.join(model_path, '*' + extension))
            if len(filenames) == 0:
                return None
            sorted_key = lambda t: os.stat(t).st_mtime
            filenames = sorted(filenames, key=sorted_key, reverse=True)
            model_path = filenames[0]
    return model_path


def get_run_name(run_tag=None, add_hostname=False):
    """A unique name for each run 
    
    References:
        https://github.com/ShuLiu1993/PANet/blob/master/lib/utils/misc.py
    """
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%y%m%d_%H%M%S')
    if (run_tag is None) or (run_tag.strip() == ''):
        run_name = f'{time_str}'
    else:
        run_name = f'{run_tag.strip()}_{time_str}'
    if add_hostname:
        run_name += f'@{socket.gethostname()}'
    return run_name
 

def get_run_save_dir(save_dir, run_tag=None, add_hostname=True, make_dir=True):
    run_name = get_run_name(run_tag, add_hostname)
    run_save_dir = os.path.join(os.path.expanduser(save_dir), run_name)
    if make_dir:
        os.makedirs(run_save_dir, exist_ok=True)
    return run_save_dir


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
            
    References:
        https://www.zhihu.com/question/406970101
        https://pytorch.org/docs/stable/notes/randomness.html
        https://pytorch.org/docs/stable/generated/torch.manual_seed.html
        https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed.html
        https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed_all.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    

@dataclass
class ModelParameterInfo:
    order: int
    name: str
    requires_grad: bool
    dtype: torch.dtype
    shape: torch.Size
    initialized: Optional[bool]
    min_val: float
    max_val: float
    
    
def convert_parameter_infos_to_markdown_table(parameter_infos: List[ModelParameterInfo]) -> khandy.MarkdownTable:
    headers = ['No.', 'name', 'requires_grad', 'dtype', 'shape', 'initialized', 'min_val', 'max_val']
    align_types = [khandy.MarkdownTableAlignType.DEFAULT for _ in range(len(headers))]
    rows = []
    for parameter_info in parameter_infos:
        initialized = 'N/A' if parameter_info.initialized is None else str(parameter_info.initialized)
        one_row = [str(parameter_info.order), str(parameter_info.name), 
                   str(parameter_info.requires_grad), str(parameter_info.dtype), 
                   str(tuple(parameter_info.shape)), initialized, 
                   f'{parameter_info.min_val:.5f}', f'{parameter_info.max_val:.5f}']
        rows.append(one_row)
    return khandy.MarkdownTable(headers, align_types, rows)


def get_model_parameter_infos(model: torch.nn.Module, missing_keys=None, convert_to_table=True
                              ) -> Union[List[ModelParameterInfo], khandy.MarkdownTable]:
    parameter_infos = []
    for k, (name, param) in enumerate(model.named_parameters()):
        if missing_keys is None:
            initialized = None
        else:
            initialized =  name not in missing_keys
        parameter_info = ModelParameterInfo(k+1, name, param.requires_grad, param.dtype, 
                                            param.shape, initialized, param.min(), param.max())
        parameter_infos.append(parameter_info)
    if convert_to_table:
        return convert_parameter_infos_to_markdown_table(parameter_infos)
    else:
        return parameter_infos

