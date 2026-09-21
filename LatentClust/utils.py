import torch
import numpy as np
import os

def int_tuple(s):
    return tuple(int(i) for i in s.split(','))

def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def dic2cuda(env_data):
    for key in env_data:
        env_data[key] = env_data[key].cuda()
    return env_data

def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    # (pre_len, num_sample, batch, 2)
    rel_traj = rel_traj.permute(2,1, 0, 3)
    displacement = torch.cumsum(rel_traj, dim=2)
    # start_pos = torch.unsqueeze(start_pos, dim=2)
    abs_traj = displacement + start_pos[:,None,None]
    return abs_traj.permute(2,1, 0, 3)

def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("\\")[:-1] #这里本地和服务器有区别，上载到服务器运行的话用反斜杠/
    _dir = "\\".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)