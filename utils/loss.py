import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parameter as Param
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as tf

from pthutils import tensorToVar, extract_patches
import matplotlib.pyplot as plt
from time import sleep

def total_variation(x):
    """
    Total Variation Loss.
    """
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
            ) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


def feature_mse_loss_func(x, y, vgg_model, mask=None, layer=['r51']):
    """
    Feature loss define by vgg layer.
    """
    if mask is not None:
        x = MaskGrad(mask)(x)
    x_feat = vgg_model(x, layer)
    y_feat = vgg_model(y, layer)
    loss = sum([nn.MSELoss()(a, b) for a, b in zip(x_feat, y_feat)])
    return loss 


def feature_mrf_loss_func(x, y, vgg_model=None, layer=[], match_img_vgg=[], topk=1):
    assert isinstance(match_img_vgg, list), 'Parameter match_img_vgg should be a list'
    mrf_crit = MRFLoss(topk=topk)
    loss = 0.
    if len(layer) == 0 or layer[0] == 'r11' or layer[0] == 'r12':
        mrf_crit.patch_size = (5, 5)
        mrf_crit.filter_patch_stride = 4
    if len(layer) == 0:
        return mrf_crit(x, y)
    x_feat = vgg_model(x, layer)
    y_feat = vgg_model(y, layer)
    match_img_feat = [vgg_model(m, layer) for m in match_img_vgg]
    if len(match_img_vgg) == 0:
        for pred, gt in zip(x_feat, y_feat):
            loss += mrf_crit(pred, gt)
    elif len(match_img_vgg) == 1:
        for pred, gt, match0  in zip(x_feat, y_feat, match_img_feat[0]):
            loss += mrf_crit(pred, gt, [match0])
    elif len(match_img_vgg) == 2:
        for pred, gt, match0, match1 in zip(x_feat, y_feat, match_img_feat[0], match_img_feat[1]):
            loss += mrf_crit(pred, gt, [match0, match1])
    return loss


def gm_loss_func(x, y, vgg_model, layer=[]):
    pass

class RCLoss(nn.Module):
    """
    Region Covariances Loss.
    """
    def __init__(self, patch_size=(3, 3), patch_stride=1):
        super(RCLoss, self).__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def get_region_cov(self, tensor):
        tensor = tensor.contiguous()
        patch_tensor = extract_patches(tensor, self.patch_size, self.patch_stride)
        B, patch_num, C, psz, psz = patch_tensor.shape
        patch_tensor = patch_tensor.contiguous().view(B, patch_num, C, -1)
        patch_tensor = patch_tensor - torch.mean(patch_tensor, -1, keepdim=True)
        pgmm = torch.matmul(patch_tensor.transpose(2, 3), patch_tensor)
        return pgmm

    def forward(self, pred, target):
        pred = self.get_region_cov(pred)
        target = self.get_region_cov(target)
        return nn.MSELoss()(pred, target)
        

class MaskGrad(torch.autograd.Function):
    """
    Mask backward gradient.
    """
    def __init__(self, mask):
        self.mask = mask

    def forward(self, x):
        return x.view_as(x) # Must change x, otherwise backward won't be called

    def backward(self, grad_output):
        return grad_output * self.mask 


class MRFLoss(nn.Module):
    """
    Feature level patch matching loss.
    """
    def __init__(self, patch_size=(3, 3), filter_patch_stride=1, compare_stride=1, topk=1):
        super(MRFLoss, self).__init__()
        self.crit = nn.MSELoss()
        self.patch_size = patch_size
        self.compare_stride = compare_stride
        self.filter_patch_stride = filter_patch_stride
        self.topk = topk

    def best_match(self, x1, x2):
        """
        x1: content feature, (B, C, H, W)
        x2: style patches, (B, nH*nW, c, patch_size, patch_size)
        """
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=2)
        match = []
        dist_func = nn.Conv2d(x1.size(1), x2.size(1), (x2.size(3), x2.size(4)), stride=self.compare_stride, bias=False)
        if torch.cuda.is_available():
            dist_func.cuda()
        dist_func.eval()
        for i in range(x1.size(0)):
            dist_func.weight.data = x2[i].squeeze().data
            cosine_dist = dist_func(x1[i].unsqueeze(0))
            tmp_match = torch.max(cosine_dist, dim=1, keepdim=False)[1]
            match.append(tmp_match.squeeze().view(-1).data)
        return match

    def best_topk_match(self, x1, x2):
        """
        Best topk match.
        x1: reference feature, (B, C, H, W)
        x2: topk candidate feature patches, (B*topk, nH*nW, c, patch_size, patch_size)
        """
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=2)
        k_match, spatial_match = [], []
        dist_func = nn.Conv2d(x1.size(1), x2.size(1), (x2.size(3), x2.size(4)), stride=self.compare_stride, bias=False)
        if torch.cuda.is_available():
            dist_func.cuda()
        dist_func.eval()
        for i in range(x1.size(0)):
            tmp_value, tmp_idx = [], []
            for j in range(self.topk):
                dist_func.weight.data = x2[i*self.topk + j].squeeze().data
                cosine_dist = dist_func(x1[i].unsqueeze(0))
                max_value, max_idx = torch.max(cosine_dist, dim=1, keepdim=False)
                tmp_value.append(max_value)
                tmp_idx.append(max_idx)
            topk_value  = torch.stack(tmp_value)
            _, k_idx    = torch.max(topk_value, dim=0, keepdim=False)
            spatial_idx = torch.stack(tmp_idx)
            k_match.append(k_idx.squeeze().view(-1).data)
            spatial_match.append(spatial_idx.squeeze(1).view(spatial_idx.shape[0], -1).data)
        return torch.stack(k_match), torch.stack(spatial_match)

    def forward(self, pred_style, target_style, match=[]):
        """
        pred_style: feature of predicted image 
        target_style: target style feature
        match: images used to match pred_style with target style 

        switch(len(match)):
            case 0: matching is done between pred_style and target_style
            case 1: matching is done between match[0] and target style
            case 2: matching is done between match[0] and match[1]
        """
        assert isinstance(match, list), 'Parameter match should be a list'
        target_style_patches = extract_patches(target_style, self.patch_size, self.filter_patch_stride)
        pred_style_patches = extract_patches(pred_style, self.patch_size, self.compare_stride)

        bk, nhnw, c, psz, psz = target_style_patches.shape

        if len(match) == 0:
            k_best_match, spatial_best_match = self.best_topk_match(pred_style, target_style_patches)
        elif len(match) == 1:
            k_best_match, spatial_best_match = self.best_topk_match(match[0], target_style_patches)
        elif len(match) == 2:
            match_patches = extract_patches(match[1], self.patch_size, self.filter_patch_stride)
            k_best_match, spatial_best_match = self.best_topk_match(match[0], match_patches)

        pred_shape = pred_style_patches.size()
        new_topk_target_style_patches = tensorToVar(torch.zeros(pred_shape[0]*self.topk,
                                                pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])) 
        spatial_best_match = spatial_best_match.view(pred_shape[0]*self.topk, -1)
        for i in range(pred_shape[0]*self.topk):
            new_topk_target_style_patches[i] = target_style_patches[[i], spatial_best_match[i]]
        new_topk_target_style_patches = new_topk_target_style_patches.view(pred_shape[0], self.topk,
                                                            pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4]) 
        new_target_style_patches = tensorToVar(torch.zeros(pred_shape)) 
        for i in range(k_best_match.shape[0]):
            for j in range(k_best_match.shape[1]):
                new_target_style_patches[i, j] = new_topk_target_style_patches[i, k_best_match[i, j], j]
        # Visulize new_target_style_patches
        #  B, nHnW, c, _, _ = new_target_style_patches.size()
        #  print(new_target_style_patches.size())
        #  feature_map = torch.mean(new_target_style_patches.view(B, nHnW, c, -1), -1)
        #  print(feature_map.size())
        #  feature_map = feature_map.view(B, np.sqrt(nHnW).astype(int), np.sqrt(nHnW).astype(int), c)
        #  for i in range(c):
            #  tmp_feature = feature_map[0, :, :, i].data.cpu().unsqueeze(0)
            #  print(tmp_feature.shape, torch.max(tmp_feature), torch.min(tmp_feature))
            #  tmp_img = tf.to_pil_image(tmp_feature)
            #  tmp_img.save('./tmp_img/{}.jpg'.format(i))
            #  print(tmp_img.size)
            #  plt.imshow(, cmap='gray')
            #  plt.waitforbuttonpress()
            #  sleep(30)
        #  exit()

        return self.crit(pred_style_patches, new_target_style_patches)


        
        

