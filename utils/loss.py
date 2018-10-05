import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parameter as Param
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as tf

from .utils import tensorToVar, extract_patches
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

    def get_new_style_map(self):
        # Visulize new_target_style_patches
        B, nHnW, c, _, _ = self.new_style_feature.size()
        feature_map = torch.mean(self.new_style_feature.view(B, nHnW, c, -1), -1)
        feature_map = feature_map.view(B, np.sqrt(nHnW).astype(int), np.sqrt(nHnW).astype(int), c)
        feature_map = feature_map.permute(0, 3, 1, 2)
        return feature_map

    def get_pixel_match(self, topk_ref_style):
        topk_style_patches = extract_patches(topk_ref_style, (12, 12), 4)
        pred_shape = list(topk_style_patches.size())
        pred_shape[0] = 1
        new_topk_target_style_patches = tensorToVar(torch.zeros(pred_shape[0]*self.topk,
                                                pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])) 
        self.spatial_best_match = self.spatial_best_match.view(pred_shape[0]*self.topk, -1)
        for i in range(pred_shape[0]*self.topk):
            new_topk_target_style_patches[i] = topk_style_patches[[i], self.spatial_best_match[i]]
        new_topk_target_style_patches = new_topk_target_style_patches.view(pred_shape[0], self.topk,
                                                            pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4]) 
        new_target_style_patches = tensorToVar(torch.zeros(pred_shape)) 
        for i in range(self.k_best_match.shape[0]):
            for j in range(self.k_best_match.shape[1]):
                new_target_style_patches[i, j] = new_topk_target_style_patches[i, self.k_best_match[i, j], j]
        B, nHnW, c, _, _ = new_target_style_patches.shape
        nH = int(np.sqrt(nHnW))
        pix_vis = new_target_style_patches[:, :, :, 4:8, 4:8].squeeze()
                
        pix_vis = pix_vis.permute(1, 0, 2, 3).contiguous()
        pix_vis = pix_vis.view(3, nH, nH, 4, 4)
        pix_vis = pix_vis.permute(0, 1, 3, 2, 4).contiguous()
        pix_vis = pix_vis.view(3, nH*4, nH*4)
        return pix_vis.unsqueeze(0)

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

        self.k_best_match = k_best_match
        self.spatial_best_match = spatial_best_match

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
        self.new_style_feature = new_target_style_patches
        loss = self.crit(pred_style_patches, new_target_style_patches)
        return loss 


        
        

