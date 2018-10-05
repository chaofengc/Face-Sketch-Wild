import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import dlib

from .utils import tensorToVar

def read_img_var(img_path, color=1, size=None):
    """
    Read image and convert it to Variable in 0~255.
    Args:
        img_path: str, test image path
        size: tuple, output size (1, C, W, H)
    """
    if color:
        img = Image.open(img_path).convert('RGB') 
    else:
        img = Image.open(img_path).convert('L') 
    if size is not None:
        img = transforms.functional.resize(img, size)
    return tensorToVar(transforms.functional.to_tensor(img)).unsqueeze(0) * 255  

def read_sketch_var(img_path, color=1, size=None, addxy=1, DoG=1):
    """
    Read image and convert it to Variable.
    Args:
        img_path: str, test image path
        size: tuple, output size (W, H)
    """
    img = Image.open(img_path).convert('L') 
    face_img = transforms.functional.resize(img, size)
    return tensorToVar(transforms.functional.to_tensor(face_img)) * 255

def read_imgAB_var(img_path, AB=0, size=None):
    """
    Read RGB image, resize to given size and convert to Variable.
    Args:
        img_path: str, test image path
        AB: read image A or image B
        size: tuple, output size(W, H)
    """
    img = Image.open(img_path).convert('RGB')
    
    width, height = img.size
    img1 = img.crop((0, 0, np.floor(width / 2).astype('int'), height))
    img2 = img.crop((np.ceil(width / 2).astype('int'), 0, width, height))
    img = img2 if AB else img1 

    if size:
        img = transforms.functional.resize(img, size)
    img_tensor = transforms.functional.to_tensor(img)
    return tensorToVar(img_tensor) * 255
   
def save_var_img(var, save_path=None, size=None):
    """
    Post processing output Variable.
    Args:
        var: Variable, (1, C, H, W)
    """
    out = var.squeeze().data.cpu().numpy()
    out[out>255] = 255  
    out[out<0]   = 0
    if len(out.shape) > 2:
        out = out.transpose(1, 2, 0)
    #  out = transforms.to_pil_image(out, mode='L')
    out = Image.fromarray(out.astype(np.uint8)).convert('RGB')
    if size:
        out = transforms.functional.resize(out, size)
    if save_path:
        out.save(save_path)
    return out

def subtract_mean_batch(batch, img_type='face', sketch_mean_shift=0):
    """
    Convert image batch to BGR and subtract imagenet mean
    Batch Size: (B, C, H, W), RGB
    Convert BGR to gray by: [0.114, 0.587, 0.299]
    """
    vgg_mean_bgr = np.array([103.939, 116.779, 123.680]) 
    sketch_mean = np.array([np.dot(vgg_mean_bgr, np.array([0.114, 0.587, 0.299]))]*3)
    if img_type == 'face':
        mean_bgr = vgg_mean_bgr
    elif img_type == 'sketch':
        mean_bgr = sketch_mean + sketch_mean_shift

    batch = batch[:, [2, 1, 0], :, :]
    batch = batch - tensorToVar(torch.Tensor(mean_bgr)).view(1, 3, 1, 1) 
    return batch

