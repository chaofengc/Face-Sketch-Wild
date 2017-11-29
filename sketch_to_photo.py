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
from utils.face_sketch_data import *
from models.sketch_net_v1 import SketchNetV1
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
    arg_parser.add_argument('--pre-epochs', type=int, default=50, help='Pretrain epochs to generate content image')
    arg_parser.add_argument('--tune-epochs', type=int, default=50, help='Fine tune epochs to generate sketch')
    arg_parser.add_argument('--weight-root', type=str, default='./weight', help='Weight saving path')
    arg_parser.add_argument('--vgg19-weight', type=str, default='/home/cfchen/pytorch_models/vgg_conv.pth',
                                                        help='Pretrained vgg19 weight path')
    arg_parser.add_argument('--addxy', type=int, default=0, help="Add (x, y) coordinate to input or not")
    arg_parser.add_argument('--DoG', type=int, default=0, help="Add DoG to input or not")
    arg_parser.add_argument('--model-version', type=int, default=1, help="Which model to use")
    arg_parser.add_argument('--norm', type=str, default='IN', help="Instance(IN) normalization or batch(BN) normalization")
    arg_parser.add_argument('--weight', type=float, nargs=3, default=[1.0, 0, 1e-5], help="MRF weight, content loss weight and total variation weight")
    arg_parser.add_argument('--other', type=str, default='', help="Other information")
    
    arg_parser.add_argument('--test-img', type=str, default='', help='Test image path')
    arg_parser.add_argument('--result-root', type=str, default='./result', help='Result saving directory')
    arg_parser.add_argument('--test-pre-model', type=str, default='', help='Test pretrain model weight path')
    arg_parser.add_argument('--test-tune-model', type=str, default='', help='Test tune model weight path')
    arg_parser.add_argument('--resume', type=int, default=0, help='Resume training or not')
    #  arg_parser.print_help()
    return arg_parser.parse_args()


def train(model, args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 4}
    else:
        kwargs = {}
    mrf_loss_layers = ['r31', 'r41']

    transform_list = [Rescale((248, 200)), ColorJitter(0.5, 0.5, 0.5, 0.3, 0.5),
                      CalDoG(), AddXY(), ToTensor()]
    trans_filter = [1, 0, args.DoG, args.addxy, 1]
    transform = transforms.Compose(list(itertools.compress(transform_list, trans_filter)))

    face_sketch_data = FaceSketchDataset(args.train_data, transform)
    data_loader      = DataLoader(face_sketch_data, shuffle=True, batch_size=args.batch_size, **kwargs)

    vgg19_model = vgg19(args.vgg19_weight, type='normal') 
    
    params = list(model.parameters())
    optimizer = Adam(params, args.lr)
    mse_crit = nn.MSELoss()
    mrf_crit = MRFLoss() 

    save_weight_dir = '{}-{}-lr{:.4f}-weight-{:.1e}-{:.1e}-{:.1e}-pre{:02d}-tune{:02d}-xy{}-DoG{}{}'.format(
                        type(model).__name__, args.norm, args.lr, args.weight[0], args.weight[1], args.weight[2], 
                        args.pre_epochs, args.tune_epochs, 
                        args.addxy, args.DoG, args.other) 
    save_weight_path = os.path.join(args.weight_root, save_weight_dir)
    if not os.path.exists(save_weight_path):
        os.mkdir(save_weight_path)

    if args.resume:
        for root, dirs, files in os.walk(save_weight_path):
            last_train_weight = os.path.join(root, sorted(files)[-1])
            model.load_state_dict(torch.load(last_train_weight))

    for e in xrange(args.pre_epochs + args.tune_epochs):
        model.train()

        if e < args.pre_epochs:
            save_weight_name = "pretrain-{:03d}.pth".format(e)
        else:
            save_weight_name = "tune-{:03d}.pth".format(e - args.pre_epochs)
        sample_count = 0 
        for batch_idx, batch_data in enumerate(data_loader):
            start = time()
            train_X, sketch_gt = batch_data['face'], batch_data['sketch']
            train_X, sketch_gt = tensorToVar(train_X), tensorToVar(sketch_gt).unsqueeze(1)
            end = time()
            data_time = end - start
            sample_count += train_X.size(0)

            start = time()
            photo_pred = model(sketch_gt)

            sketch_gt_vgg = sketch_gt.expand_as(train_X)
            photo_pred_vgg = photo_pred.expand_as(train_X)
            train_X_vgg = img_process.subtract_imagenet_mean_batch(train_X)
            sketch_gt_vgg = img_process.subtract_imagenet_mean_batch(sketch_gt_vgg)
            photo_pred_vgg = img_process.subtract_imagenet_mean_batch(photo_pred_vgg)

            if args.pre_epochs == 0:    # No pretrain
                photo_feat = vgg19_model(train_X_vgg, mrf_loss_layers)
                sketch_feat = vgg19_model(sketch_gt_vgg, mrf_loss_layers)
                pred_feat = vgg19_model(photo_pred_vgg, mrf_loss_layers)
                loss = 0.
                for (cf, sf, pf) in zip(sketch_feat, photo_feat, pred_feat):  
                    loss += mrf_crit(cf, sf, pf) 
            elif e < args.pre_epochs:
                loss = mse_crit(photo_pred, train_X)
            else:
                photo_feat = vgg19_model(photo_pred_vgg, mrf_loss_layers)
                sketch_feat = vgg19_model(sketch_gt_vgg, mrf_loss_layers)
                loss = 0.
                for (cf, sf) in zip(sketch_feat, photo_feat):  
                    loss += mrf_crit(cf, sf) 
            
            # add content loss and total variation loss
            loss = args.weight[0] * loss + args.weight[1] * content_loss(photo_pred_vgg, train_X_vgg, 
                    vgg19_model) + args.weight[2] * total_variation(photo_pred)  
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time()
            train_time = end - start

            msg = "{:%Y-%m-%d %H:%M:%S}\tEpoch [{:03d}/{:03d}-{:03d}]\tBatch [{:03d}/{:03d}]\tData: {:.2f}  Train: {:.2f}\tLoss: {:.4f}".format(
                            datetime.now(), 
                            e, args.pre_epochs, args.tune_epochs, sample_count, len(face_sketch_data),
                            data_time, train_time, loss.data[0])
            print msg
            log_file = open(os.path.join(save_weight_path, 'log.txt'), 'a+')
            log_file.write(msg + '\n')
            log_file.close()

        torch.save(model.state_dict(), os.path.join(save_weight_path, save_weight_name))


def test(model, args):
    model.eval()
    sketch_img = img_process.read_sketch_var(args.test_img, size=(248, 200), addxy=args.addxy, DoG=args.DoG)
    sketch_img = sketch_img.unsqueeze(0)

    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)
        os.mkdir(os.path.join(args.result_root, 'photo'))
    
    save_weight_dir = '{}-{}-lr{:.4f}-weight-{:.1e}-{:.1e}-{:.1e}-pre{:02d}-tune{:02d}-xy{}-DoG{}{}'.format(
                        type(model).__name__, args.norm, args.lr, args.weight[0], args.weight[1], args.weight[2], 
                        args.pre_epochs, args.tune_epochs, 
                        args.addxy, args.DoG, args.other) 
    save_weight_path = os.path.join(args.weight_root, save_weight_dir)

    if args.pre_epochs > 0:
        model.load_state_dict(torch.load(os.path.join(save_weight_path, args.test_pre_model)))
        sketch_pred = model(face_img)
        content_save_path = os.path.join(args.result_root, 'content', os.path.basename(args.test_img))
        img_process.save_var_img(sketch_pred, content_save_path, size=(200, 250))
        print 'content saved in {}'.format(content_save_path)

    model.load_state_dict(torch.load(os.path.join(save_weight_path, args.test_tune_model)))
    face_pred = model(sketch_img)
    face_save_path = os.path.join(args.result_root, 'photo', os.path.basename(args.test_img))
    img_process.save_var_img(face_pred, face_save_path, size=(250, 200))

    print('face saved in {}'.format(face_save_path))

if __name__ == '__main__':
    gm=GPUManager()
    torch.cuda.set_device(gm.auto_choice())
    args = cmd_option()

    in_channels = 1 + args.DoG + args.addxy * 2
    model_list = [SketchNetV1(in_channels=in_channels, out_channels=3, norm=args.norm),
                  SketchNetV2(in_channels=in_channels, out_channels=3)]
    model = model_list[args.model_version-1] 
    if torch.cuda.is_available():
        model.cuda()

    if args.train_eval == 'train':
        train(model, args)
    else:
        test(model, args)


    
