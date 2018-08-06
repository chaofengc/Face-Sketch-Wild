import torch
from torch.autograd import Variable
import os

def mkdirs(dirs):
    if isinstance(dirs, list):
        for i in dirs:
            if not os.path.exists(i):
                os.makedirs(i)
    elif isinstance(dirs, str):
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    else:
        raise Exception('dirs should be list or string.')


def to_device(tensor):
    """
    Move tensor to device. If GPU is is_available, move to GPU.
    """
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def tensorToVar(tensor):
    """
    Convert a tensor to Variable
    If cuda is avaible, move to GPU
    """
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


def extract_patches(img, patch_size=(3, 3), stride=(1, 1)):
    """
    Divide img into overlapping patches with stride = 1
    img: (b, c, h, w)
    output patches: (b, nH*nW, c, patch_size)
    """
    assert type(patch_size) in [int, tuple], 'patch size should be int or tuple int'
    assert type(stride) in [int, tuple], 'stride size should be int or tuple int'
    if type(stride) is int:
        stride = (stride, stride)
    if type(patch_size) is int:
        patch_size = (patch_size, patch_size)
    patches = img.unfold(2, patch_size[0], stride[0]).unfold(3, patch_size[1], stride[1]) 
    patches = patches.contiguous().view(img.shape[0], img.shape[1], -1, patch_size[0], patch_size[1])
    patches = patches.transpose(1, 2)
    return patches

