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

from pthutils import tensorToVar 
from gpu_manager import GPUManager
from utils.pix2pix_data import *
from models.sketch_net_v0 import SketchNetV1
from models.sketch_net_v2 import SketchNetV2
from models.vgg19 import vgg19 
from utils.loss import MRFLoss, total_variation, content_loss
from utils import img_process 

def cmd_option():
    arg_parser = argparse.ArgumentParser(description='CMD arguments for the face sketch network')
    arg_parser.add_argument('train_eval', type=str, default='train', help='Train or eval')
    arg_parser.add_argument('--gpuid', type=str, default='0', help='On which gpu to run the program')
    arg_parser.add_argument('--train-data', type=str, default="./data", help="Train data dir root")
    arg_parser.add_argument('--seed', type=int, default=123, help='Random seed for training')
    arg_parser.add_argument('--batch-size', type=int, default=12, help='Train batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    arg_parser.add_argument('--pre-iter', type=int, default=1000, help='Pretrain iterations to generate content image')
    arg_parser.add_argument('--epochs', type=int, default=50, help='Training epochs to generate sketch')
    arg_parser.add_argument('--weight-root', type=str, default='./weight', help='Weight saving path')
    arg_parser.add_argument('--vgg19-weight', type=str, default='/home/cfchen/pytorch_models/vgg_conv.pth',
                                                        help='Pretrained vgg19 weight path')
    arg_parser.add_argument('--model-version', type=int, default=1, help="Which model to use")
    arg_parser.add_argument('--norm', type=str, default='IN', help="Instance(IN) normalization or batch(BN) normalization")
    arg_parser.add_argument('--weight', type=float, nargs=3, default=[1.0, 1e-2, 1e-5], help="MRF weight, content loss weight and total variation weight")
    arg_parser.add_argument('--other', type=str, default='', help="Other information")
    
    arg_parser.add_argument('--test-img', type=str, default='', help='Test image path')
    arg_parser.add_argument('--result-root', type=str, default='./result', help='Result saving directory')
    arg_parser.add_argument('--test-pre-model', type=str, default='', help='Test pretrain model weight path')
    arg_parser.add_argument('--test-tune-model', type=str, default='', help='Test tune model weight path')
    arg_parser.add_argument('--resume', type=int, default=0, help='Resume training or not')
    #  arg_parser.print_help()
    return arg_parser.parse_args()


def train(model, args, save_weight_dir, save_weight_path):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 4}
    else:
        kwargs = {}
    mrf_loss_layers = ['r31', 'r41']

    transform_list = [
                      CropAndResize(224),
                      #  Rescale((256, 256)), 
                      ToTensor()]
    trans_filter = [1, 1]
    transform = transforms.Compose(list(itertools.compress(transform_list, trans_filter)))

    p2p_data = Pix2PixDataset(args.train_data, transform)
    data_loader = DataLoader(p2p_data, shuffle=True, batch_size=args.batch_size, **kwargs)

    vgg19_model = vgg19(args.vgg19_weight, type='normal') 
    
    params = list(model.parameters())
    optimizer = Adam(params, args.lr)
    mse_crit = nn.MSELoss()
    mrf_crit = MRFLoss() 
    
    if args.resume:
        for root, dirs, files in os.walk(save_weight_path):
            last_train_weight = os.path.join(root, sorted(files)[-1])
            model.load_state_dict(torch.load(last_train_weight))

    for e in range(args.epochs):
        model.train()

        loss_weight = [1.0, 0, args.weight[2]]  # Pretrain loss weight

        sample_count = 0 
        for batch_idx, batch_data in enumerate(data_loader):
            start = time()
            train_img, gt_img = batch_data[0], batch_data[1]
            train_img, gt_img = tensorToVar(train_img), tensorToVar(gt_img)
            end = time()
            data_time = end - start
            sample_count += train_img.size(0)

            start = time()
            photo_pred = model(train_img)

            #  train_img_vgg = img_process.subtract_imagenet_mean_batch(train_img)
            gt_img_vgg = img_process.subtract_imagenet_mean_batch(gt_img)
            photo_pred_vgg = img_process.subtract_imagenet_mean_batch(photo_pred)
            gt_feat = vgg19_model(gt_img_vgg, mrf_loss_layers)
            pred_feat = vgg19_model(photo_pred_vgg, mrf_loss_layers)

            mse_loss = mse_crit(photo_pred, gt_img)
            tv_loss = total_variation(photo_pred)

            mrf_loss = 0.
            for (pf, gf) in zip(pred_feat, gt_feat):
                mrf_loss += mrf_crit(pf, gf)

            loss_list = [a * b for a, b in zip(loss_weight, [mse_loss, mrf_loss, tv_loss])]
            loss = sum(loss_list)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time()
            train_time = end - start

            msg = "{:%Y-%m-%d %H:%M:%S}\tEpoch [{:03d}/{:03d}-{:03d}]\tBatch [{:03d}/{:03d}]\tData: {:.2f}  Train: {:.2f}\tLoss: {:.4f} [{:.4f}, {:.4f}, {:.4f}]".format(
                            datetime.now(), 
                            e, args.pre_epochs, args.tune_epochs, sample_count, len(p2p_data),
                            data_time, train_time, loss.data[0], *[x.data[0] for x in loss_list])
            print msg
            log_file = open(os.path.join(save_weight_path, 'log.txt'), 'a+')
            log_file.write(msg + '\n')
            log_file.close()
        
        if (e + 1)*(batch_idx + 1) < args.pre_iter:
            save_weight_name = "pretrain-{:03d}.pth".format(e)
        else:
            save_weight_name = "tune-{:03d}.pth".format(e)
            loss_weight = args.weight

        torch.save(model.cpu().state_dict(), os.path.join(save_weight_path, save_weight_name))


def test(model, args, save_weight_dir, save_weight_path):
    size = None 
    model.eval()
    sketch_img = img_process.read_imgAB_var(args.test_img, AB=0, size=size)
    sketch_img = sketch_img.unsqueeze(0)

    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)
        os.mkdir(os.path.join(args.result_root, 'photo'))
        os.mkdir(os.path.join(args.result_root, 'content'))
    
    if args.pre_epochs > 0:
        model.load_state_dict(torch.load(os.path.join(save_weight_path, args.test_pre_model)))
        sketch_pred = model(sketch_img)
        content_save_path = os.path.join(args.result_root, 'content', os.path.basename(args.test_img))
        img_process.save_var_img(sketch_pred, content_save_path, size=size)
        print 'content saved in {}'.format(content_save_path)

    model.load_state_dict(torch.load(os.path.join(save_weight_path, args.test_tune_model)))
    face_pred = model(sketch_img)
    face_save_path = os.path.join(args.result_root, 'photo', os.path.basename(args.test_img))
    img_process.save_var_img(face_pred, face_save_path, size=size)

    print('Image saved in {}'.format(face_save_path))

if __name__ == '__main__':
    gm=GPUManager()
    torch.cuda.set_device(gm.auto_choice())
    args = cmd_option()

    in_channels = 3 
    model_list = [SketchNetV1(in_channels=in_channels, out_channels=3, norm=args.norm),
                  SketchNetV2(in_channels=in_channels, out_channels=3)]
    model = model_list[args.model_version-1] 
    if torch.cuda.is_available():
        model.cuda()

    save_weight_dir = 'pix2pix-{}-{}-{}-lr{:.4f}-weight-{:.1e}-{:.1e}-{:.1e}-pre{:02d}-epoch{:02d}-{}'.format(
                        args.train_data.split('/')[-2], type(model).__name__, args.norm, args.lr, args.weight[0], args.weight[1], args.weight[2], 
                        args.pre_iter, args.epochs, 
                        args.other) 
    save_weight_path = os.path.join(args.weight_root, save_weight_dir)
    if not os.path.exists(save_weight_path):
        os.mkdir(save_weight_path)

    if args.train_eval == 'train':
        train(model, args, save_weight_dir, save_weight_path)
    else:
        test(model, args, save_weight_dir, save_weight_path)


    

