import os
import glob
import socket
import random
import datetime
import functools

import torch
import khandy
import numpy as np


__all__ = ['sum_tensor_list', 'get_latest_model_path', 'get_run_name', 
           'get_run_save_dir', 'check_state_dict_keys', 'set_random_seed']


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


def get_run_name(run_tag=None):
    """A unique name for each run 
    
    References:
        https://github.com/ShuLiu1993/PANet/blob/master/lib/utils/misc.py
    """
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%y%m%d_%H%M%S')
    if run_tag is not None:
        run_name = '{}_{}@{}'.format(run_tag, time_str, socket.gethostname())
    else:
        run_name = '{}@{}'.format(time_str, socket.gethostname())
    return run_name
 

def get_run_save_dir(save_dir, run_tag=None, make_dir=True):
    run_name = get_run_name(run_tag)
    run_save_dir = os.path.join(os.path.expanduser(save_dir), run_name)
    if make_dir:
        os.makedirs(run_save_dir, exist_ok=True)
    return run_save_dir


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
    
