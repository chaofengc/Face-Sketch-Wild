from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import argparse
import os
import numpy as np
import cv2 as cv
from time import time
from datetime import datetime
import itertools
import copy
from glob import glob

from pthutils import tensorToVar 
from gpu_manager import GPUManager
from utils.pix2pix_data import *
from utils.face_sketch_data import FaceSketchDataset
from utils.pair_transform import *
from models.sketch_net_v1 import SketchNetV1
from models.sketch_net_v2 import SketchNetV2
from models.vgg19 import vgg19 
from utils.loss import * 
from utils import img_process 

def cmd_option():
    arg_parser = argparse.ArgumentParser(description='CMD arguments for the face sketch network')
    arg_parser.add_argument('train_eval', type=str, default='train', help='Train or eval')
    arg_parser.add_argument('--train-data', type=str, default="./data", help="Train data dir root")
    arg_parser.add_argument('--seed', type=int, default=123, help='Random seed for training')
    arg_parser.add_argument('--batch-size', type=int, default=12, help='Train batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    arg_parser.add_argument('--epochs', type=int, default=50, help='Training epochs to generate')
    arg_parser.add_argument('--weight-root', type=str, default='./weight', help='Weight saving path')
    arg_parser.add_argument('--vgg19-weight', type=str, default='/home/cfchen/pytorch_models/vgg_conv.pth',
                                                        help='Pretrained vgg19 weight path')
    arg_parser.add_argument('--model-version', type=int, default=1, help="Which model to use")
    arg_parser.add_argument('--norm', type=str, default='IN', help="Instance(IN) normalization or batch(BN) normalization")
    arg_parser.add_argument('--layers', type=int, nargs=5, default=[0, 0, 1, 1, 1], help="Layers used to calculate feature loss")
    arg_parser.add_argument('--weight', type=float, nargs=3, default=[1e-2, 1e0, 1e-5], help="MSE loss weight, Feature loss weight, and total variation weight")
    arg_parser.add_argument('--loss-func', type=int, default=0, help="Feature loss type: mse, mrf, gm")
    arg_parser.add_argument('--direction', type=str, default='AtoB', help="Which direction to translate image.")
    arg_parser.add_argument('--other', type=str, default='', help="Other information")
    
    arg_parser.add_argument('--test-img', type=str, default='', help='Test image path')
    arg_parser.add_argument('--result-root', type=str, default='./result', help='Result saving directory')
    arg_parser.add_argument('--test-model', type=str, default='', help='Test model weight path')
    arg_parser.add_argument('--resume', type=int, default=0, help='Resume training or not')
    #  arg_parser.print_help()
    return arg_parser.parse_args()


def train(model, args, save_weight_dir, save_weight_path):
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 4}
    else:
        torch.manual_seed(args.seed)
        kwargs = {}

    transform_list = [
                      CropAndResize(224),
                      #  Rescale((256, 256)), 
                      ColorJitter(0.5, 0.5, 0.5, 0.3, 0.5),
                      ToTensor()]
    if args.direction == 'AtoB':
        trans_filter = [1, 1, 1]
    else:
        trans_filter = [1, 0, 1]
    transform = transforms.Compose(list(itertools.compress(transform_list, trans_filter)))

    if len(glob(os.path.join(args.train_data, '*/'))):
        p2p_data = FaceSketchDataset(args.train_data, transform)
    else:
        p2p_data = Pix2PixDataset(args.train_data, transform)

    data_loader = DataLoader(p2p_data, shuffle=True, batch_size=args.batch_size, **kwargs)

    vgg19_model = vgg19(args.vgg19_weight, type='normal') 
    
    params = list(model.parameters())
    optimizer = Adam(params, args.lr)
    mse_crit = nn.MSELoss()
    
    if args.resume:
        weights = glob(os.path.join(save_weight_path, '*.pth'))
        model.load_state_dict(torch.load(sorted(weights)[-1]))

    vgg_feature_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    feature_loss_layers = list(itertools.compress(vgg_feature_layers, args.layers)) 
    feature_loss_func = [feature_mse_loss_func, feature_mrf_loss_func]
    for e in range(args.epochs):
        model.train()

        sample_count = 0 
        for batch_idx, batch_data in enumerate(data_loader):
            start = time()
            train_img, gt_img = batch_data[0], batch_data[1]
            train_img, gt_img = tensorToVar(train_img), tensorToVar(gt_img)
            if args.direction == "BtoA":
                train_img, gt_img = gt_img, train_img
            end = time()
            data_time = end - start
            sample_count += train_img.size(0)

            start = time()
            photo_pred = model(train_img)
            photo_pred = photo_pred.expand_as(train_img)

            train_img_vgg = img_process.subtract_imagenet_mean_batch(train_img)
            gt_img_vgg = img_process.subtract_imagenet_mean_batch(gt_img)
            photo_pred_vgg = img_process.subtract_imagenet_mean_batch(photo_pred)
            #  gt_feat = vgg19_model(gt_img_vgg, mrf_loss_layers)
            #  pred_feat = vgg19_model(photo_pred_vgg, mrf_loss_layers)

            mse_loss = mse_crit(photo_pred, gt_img)
            tv_loss = total_variation(photo_pred)

            if args.loss_func < 3:
                feature_loss = feature_loss_func[args.loss_func](photo_pred_vgg, gt_img_vgg, vgg19_model, feature_loss_layers)
            if args.loss_func == 3:
                feature_loss = feature_mrf_loss_func(photo_pred_vgg, gt_img_vgg, vgg19_model, feature_loss_layers, train_img_vgg)
            if args.loss_func == 4:
                feature_loss = feature_mse_loss_func(photo_pred_vgg, gt_img_vgg, vgg19_model, feature_loss_layers) + feature_mrf_loss_func(photo_pred_vgg, gt_img_vgg, vgg19_model, feature_loss_layers)

            loss_list = [a * b for a, b in zip(args.weight, [mse_loss, feature_loss, tv_loss])]
            #  loss = sum(loss_list) + feature_mse_loss_func(photo_pred_vgg, gt_img_vgg, vgg19_model, feature_loss_layers)
            loss = sum(loss_list)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time()
            train_time = end - start

            msg = "{:%Y-%m-%d %H:%M:%S}\tEpoch [{:03d}/{:03d}]\tBatch [{:03d}/{:03d}]\tData: {:.2f}  Train: {:.2f}\tLoss: {:.4f} [{:.4f}, {:.4f}, {:.4f}]".format(
                            datetime.now(), 
                            e, args.epochs, sample_count, len(p2p_data),
                            data_time, train_time, loss.data[0], *[x.data[0] for x in loss_list])
            print(msg)
            log_file = open(os.path.join(save_weight_path, 'log.txt'), 'a+')
            log_file.write(msg + '\n')
            log_file.close()
        
        save_weight_name = "epochs-{:03d}.pth".format(e)

        cpu_model = copy.deepcopy(model).cpu() 
        torch.save(cpu_model.state_dict(), os.path.join(save_weight_path, save_weight_name))


def test(model, args, save_weight_dir, save_weight_path):
    size = None 
    model.eval()
    if 'photo' not in args.test_img:
        sketch_img = img_process.read_imgAB_var(args.test_img, AB=0, size=size)
    else:
        sketch_img = img_process.read_img_var(args.test_img)
    sketch_img = sketch_img.unsqueeze(0)

    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)
        os.mkdir(os.path.join(args.result_root, 'result'))
    
    model.load_state_dict(torch.load(os.path.join(save_weight_path, args.test_model)))
    face_pred = model(sketch_img)
    face_save_path = os.path.join(args.result_root, 'result', os.path.basename(args.test_img))
    img_process.save_var_img(face_pred, face_save_path, size=size)

    print('Image saved in {}'.format(face_save_path))

if __name__ == '__main__':
    gm=GPUManager()
    torch.cuda.set_device(gm.auto_choice())
    args = cmd_option()

    in_channels = 3 
    out_channels = 3
    model_list = [SketchNetV1(in_channels=in_channels, out_channels=out_channels, norm=args.norm),
                  SketchNetV2(in_channels=in_channels, out_channels=out_channels)]
    model = model_list[args.model_version-1] 
    if torch.cuda.is_available():
        model.cuda()
    loss_func_list = ['mse', 'mrf', 'gm']
    save_weight_dir = 'pix2pix-{}-{}-{}-{}-lr{:.4f}-layers{}-loss_{}-weight-{:.1e}-{:.1e}-{:.1e}-epoch{:02d}-{}'.format(
                        args.train_data.split('/')[-2], args.direction, type(model).__name__, args.norm, args.lr, 
                        "".join(map(str, args.layers)), args.loss_func, args.weight[0], args.weight[1], args.weight[2], 
                        args.epochs, args.other) 
    #  save_weight_dir = 'pix2pix-{}-{}-{}-lr{:.4f}-layers{}-loss_{}-weight-{:.1e}-{:.1e}-{:.1e}-epoch{:02d}-{}'.format(
                        #  args.train_data.split('/')[-2], type(model).__name__, args.norm, args.lr, 
                        #  "".join(map(str, args.layers)), args.loss_func, args.weight[0], args.weight[1], args.weight[2], 
                        #  args.epochs, args.other) 
    save_weight_path = os.path.join(args.weight_root, save_weight_dir)

    if args.train_eval == 'train':
        print('Saving weight path', save_weight_path)
        if not os.path.exists(save_weight_path):
            os.mkdir(save_weight_path)
        train(model, args, save_weight_dir, save_weight_path)
    else:
        test(model, args, save_weight_dir, save_weight_path)


    

