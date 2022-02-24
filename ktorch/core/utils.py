import os
import glob
import functools

import torch

__all__ = ['sum_tensor_list', 'get_lastest_model_path']


def sum_tensor_list(tensor_list):
    """
    References:
        tf.math.add_n
    """
    return functools.reduce(lambda acc, x: acc.add_(x), tensor_list,
                            torch.zeros_like(tensor_list[0]))
                            
                            
def get_lastest_model_path(model_path):
    """
    References:
        tf.train.latest_checkpoint
    """
    if model_path is not None:
        model_path = os.path.expanduser(model_path)
        if os.path.isdir(model_path):
            filenames = glob.glob(os.path.join(model_path, '*.pth'))
            if len(filenames) == 0:
                return None
            sorted_key = lambda t: os.stat(t).st_mtime
            filenames = sorted(filenames, key=sorted_key, reverse=True)
            model_path = filenames[0]
    return model_path


