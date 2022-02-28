import os
import glob
import socket
import datetime
import functools

import torch
import khandy

__all__ = ['sum_tensor_list', 'get_latest_model_path', 'get_run_name', 
           'get_run_save_dir', 'check_state_dict_keys']


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
 

def get_run_save_dir(run_tag, save_dir, make_dir=True):
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
