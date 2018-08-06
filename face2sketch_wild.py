import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR 
from torchvision import transforms
from torchvision.utils import save_image

import argparse
import os
import numpy as np
from time import time
from datetime import datetime
import itertools
import copy
from glob import glob
import random
from PIL import Image 
import math

from gpu_manager import GPUManager
from utils.face_sketch_data import * 
from models.networks import SketchNet, DNet
from models.vgg19 import vgg19 
from utils import loss 
from utils import img_process 
from utils import search_dataset
from utils import logger
from utils import utils
from utils.metric import avg_score

def cmd_option():
    arg_parser = argparse.ArgumentParser(description='CMD arguments for the face sketch network')
    arg_parser.add_argument('train_eval', type=str, default='train', help='Train or eval')
    arg_parser.add_argument('--gpus', type=str, default='0', help='Which gpus to train the model')
    arg_parser.add_argument('--train-data', type=str, nargs='*', default=["./data/AR/train_photos", "./data/CUHK_student/train_photos", "./data/XM2VTS/train_photos", "./data/CUFSF/train_photos"], help="Train data dir root")
    arg_parser.add_argument('--train-style', type=int, nargs=4, default=[1, 1, 1, 0], help='Styles used to train')
    arg_parser.add_argument('--seed', type=int, default=123, help='Random seed for training')
    arg_parser.add_argument('--batch-size', type=int, default=6, help='Train batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    arg_parser.add_argument('--epochs', type=int, default=40, help='Training epochs to generate')
    arg_parser.add_argument('--weight-root', type=str, default='./weight', help='Weight saving path')
    arg_parser.add_argument('--vgg19-weight', type=str, default='/home/cfchen/pytorch_models/vgg_conv.pth',
                                                        help='Pretrained vgg19 weight path')
    arg_parser.add_argument('--model-version', type=int, default=1, help="Which model to use")
    arg_parser.add_argument('--Gnorm', type=str, default='IN', help="Instance(IN) normalization or batch(BN) normalization")
    arg_parser.add_argument('--Dnorm', type=str, default='None', help="Instance(IN) normalization or batch(BN) normalization")
    arg_parser.add_argument('--with-rec', type=int, default=0, help="Whether to do reconstruction")
    arg_parser.add_argument('--flayers', type=int, nargs=5, default=[0, 0, 1, 1, 1], help="Layers used to calculate feature loss")
    arg_parser.add_argument('--clayers', type=int, nargs=5, default=[0, 0, 1, 0, 0], help="Which layer to calculate content loss")
    arg_parser.add_argument('--weight', type=float, nargs=3, default=[1e0, 1e3, 1e-5], help="MSE loss weight, Feature loss weight, and total variation weight")
    arg_parser.add_argument('--direction', type=str, default='AtoB', help="Which direction to translate image.")
    arg_parser.add_argument('--topk', type=int, default=1, help="Topk image choose to match input photo")
    arg_parser.add_argument('--other', type=str, default='', help="Other information")
    arg_parser.add_argument('--ref-feature', type=str, default='./data/feature_dataset.pth')
    arg_parser.add_argument('--ref-img-list', type=str, default='./data/dataset_img_list.txt')
    
    arg_parser.add_argument('--test-dir', type=str, default='', help='Test image directory')
    arg_parser.add_argument('--result-root', type=str, default='./result', help='Result saving directory')
    arg_parser.add_argument('--test-epoch', type=int, default=10, help='Test model epoch')
    arg_parser.add_argument('--resume', type=int, default=0, help='Resume training or not')
    return arg_parser.parse_args()

def train(args):
    torch.backends.cudnn.benchmark=True
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    # -------------------- Load data ----------------------------------
    transform = transforms.Compose([
                    Rescale((224, 224)), 
                    ColorJitter(0.5, 0.5, 0.5, 0.3, 0.5),
                    ToTensor(),
        ])
    dataset = FaceDataset(args.train_data, transform=transform) 
    data_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, drop_last=True, num_workers=4) 

    # ----------------- Define networks ---------------------------------
    Gnet= SketchNet(in_channels=3, out_channels=1, norm_type=args.Gnorm)
    Dnet = DNet(norm_type=args.Dnorm)
    vgg19_model = vgg19(args.vgg19_weight) 

    gpu_ids = [int(x) for x in args.gpus.split(',')]
    if len(gpu_ids) > 0:
        Gnet.cuda()
        Dnet.cuda()
        Gnet = nn.DataParallel(Gnet, device_ids=gpu_ids) 
        Dnet = nn.DataParallel(Dnet, device_ids=gpu_ids)
        vgg19_model = nn.DataParallel(vgg19_model, device_ids=gpu_ids)

    Gnet.train()
    Dnet.train()

    if args.resume:
        weights = glob(os.path.join(args.save_weight_path, '*-*.pth'))
        weight_path = sorted(weights[-1])[:-5]
        Gnet.load_state_dict(torch.load(weight_path + 'G.pth'))
        Dnet.load_state_dict(torch.load(weight_path + 'D.pth'))

    # ---------------- set optimizer and learning rate ---------------------
    args.epochs = math.ceil(args.epochs * 1000 / len(dataset))
    args.epochs = max(int(args.epochs), 4)

    optim_G = Adam(Gnet.parameters(), args.lr)
    optim_D = Adam(Dnet.parameters(), args.lr)
    scheduler_G = MultiStepLR(optim_G, milestones=[int(1./4 * args.epochs), int(2./4 * args.epochs)], gamma=0.1)
    scheduler_D = MultiStepLR(optim_D, milestones=[int(1./4 * args.epochs), int(2./4 * args.epochs)], gamma=0.1)
    mse_crit  = nn.MSELoss()
    
    # ---------------------- Define reference styles and feature loss layers ----------        
    ref_style =['CUHK_student', 'AR', 'XM2VTS', 'CUFSF']
    ref_style_dataset = list(itertools.compress(ref_style, args.train_style))
    vgg_feature_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    feature_loss_layers = list(itertools.compress(vgg_feature_layers, args.flayers)) 

    log = logger.Logger(args.save_weight_path)

    for e in range(args.epochs):
        scheduler_G.step()
        scheduler_D.step()
        sample_count = 0 
        for batch_idx, batch_data in enumerate(data_loader):
            # ---------------- Load data -------------------
            start = time()
            train_img, train_img_org = [utils.tensorToVar(x) for x in batch_data]
            topk_sketch_img, topk_photo_img = search_dataset.find_photo_sketch_batch(
                            train_img_org, args.ref_feature, args.ref_img_list,
                            vgg19_model, dataset_filter=ref_style_dataset, topk=args.topk)
            random_real_sketch = search_dataset.get_real_sketch_batch(train_img.size(0), args.ref_img_list, dataset_filter=ref_style_dataset)
            end           = time()
            data_time     = end - start
            sample_count += train_img.size(0)

            # ---------------- Model forward -------------------
            start = time()
            fake_sketch = Gnet(train_img)           
            fake_score = Dnet(fake_sketch)
            real_score = Dnet(random_real_sketch)

            real_label = torch.ones_like(fake_score) 
            fake_label = torch.zeros_like(fake_score)

            # ----------------- Calculate loss and backward ------------------- 
            train_img_org_vgg   = img_process.subtract_mean_batch(train_img_org, 'face')
            topk_sketch_img_vgg = img_process.subtract_mean_batch(topk_sketch_img, 'sketch')
            topk_photo_img_vgg  = img_process.subtract_mean_batch(topk_photo_img, 'face')
            fake_sketch_vgg = img_process.subtract_mean_batch(fake_sketch.expand_as(train_img_org), 'sketch')

            style_loss = loss.feature_mrf_loss_func(
                                fake_sketch_vgg, topk_sketch_img_vgg, vgg19_model,
                                feature_loss_layers, [train_img_org_vgg, topk_photo_img_vgg], topk=args.topk) 
            
            tv_loss = loss.total_variation(fake_sketch)

            # GAN Loss
            adv_loss = mse_crit(fake_score, real_label) * args.weight[1]
            loss_G = style_loss * args.weight[0] + adv_loss + tv_loss * args.weight[2]
            loss_D = 0.5 * mse_crit(fake_score, fake_label) + 0.5 * mse_crit(real_score, real_label) 

            # Update parameters 
            optim_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optim_D.step()

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            end = time()
            train_time = end - start

            log.iterLogUpdate(loss_G.data[0])
            if batch_idx % 100 == 0:
                log.draw_loss_curve()

            msg = "{:%Y-%m-%d %H:%M:%S}\tEpoch [{:03d}/{:03d}]\tBatch [{:03d}/{:03d}]\tData: {:.2f}  Train: {:.2f}\tLoss: G-{:.4f}, Adv-{:.4f}, D-{:.4f}".format(
                            datetime.now(), 
                            e, args.epochs, sample_count, len(dataset),
                            data_time, train_time, *[x.data[0] for x in [loss_G, adv_loss, loss_D]])
            print(msg)
            log_file = open(os.path.join(args.save_weight_path, 'log.txt'), 'a+')
            log_file.write(msg + '\n')
            log_file.close()
        
        val(copy.deepcopy(Gnet), e, os.path.join(args.save_weight_path, 'val')) 
        save_weight_name = "epochs-{:03d}-".format(e)
        G_cpu_model = copy.deepcopy(Gnet).cpu() 
        D_cpu_model = copy.deepcopy(Dnet).cpu()
        torch.save(G_cpu_model.state_dict(), os.path.join(args.save_weight_path, save_weight_name+'G.pth'))
        torch.save(D_cpu_model.state_dict(), os.path.join(args.save_weight_path, save_weight_name+'D.pth'))

def val(model, epochs, save_val_dir):
    utils.mkdirs(save_val_dir)
    quan_result_file = open(save_val_dir + '/score.txt', 'a+')
    val_dirs = ['./data/CUFS', './data/CUFSF_crop', './data/vgg_test']
    save_val_dir = os.path.join(save_val_dir, '{:02d}'.format(epochs)) 
    utils.mkdirs(save_val_dir)
    model.eval()

    # Generate results for images
    scores = []
    for val_d in val_dirs:
        if 'CUFS' in val_d:
            img_dir = os.path.join(val_d, 'test_photos')
        else:
            img_dir = val_d
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            photo_input = img_process.read_img_var(img_path, size=(256, 256))
            sketch_pred = model(photo_input)
            save_path = os.path.join(save_val_dir, val_d.split('/')[-1], img_name)
            utils.mkdirs(os.path.join(save_val_dir, val_d.split('/')[-1]))
            img_process.save_var_img(sketch_pred.squeeze(), save_path, size=(250, 200))

        if 'vgg_test' in val_d: continue

        # SSIM/FSIM score
        ssim_score = avg_score(os.path.join(save_val_dir, val_d.split('/')[-1]), 
                os.path.join(val_d, 'test_sketches'), smooth=False, metric_name='ssim')
        fsim_score = avg_score(os.path.join(save_val_dir, val_d.split('/')[-1]), 
                os.path.join(val_d, 'test_sketches'), smooth=False, metric_name='fsim')
        scores += [ssim_score, fsim_score]

    quan_result_file.write('{:02d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(epochs, *scores))
    quan_result_file.close()

def test(model, rec_model, args, save_weight_dir, save_weight_path):
    size = (250, 200) 
    model.eval()

    if not os.path.exists(args.result_root): os.mkdir(args.result_root)
    #  save_result_dir = args.result_root
    save_result_dir = os.path.join(args.result_root, 'result_{:02d}'.format(args.test_epoch))
    if not os.path.exists(save_result_dir): os.mkdir(save_result_dir)
    
    model.load_state_dict(torch.load(os.path.join(save_weight_path, 'epochs-{:03d}-p2s.pth'.format(args.test_epoch))))
    if args.with_rec:
        rec_model.load_state_dict(torch.load(os.path.join(save_weight_path, 'epochs-{:03d}-rec.pth'.format(args.test_epoch))))

    for img_name in os.listdir(args.test_dir):
        test_img_path = os.path.join(args.test_dir, img_name)
        test_img = img_process.read_img_var(test_img_path, size=(256, 256))
        face_pred = model(test_img)
        if face_pred.shape != test_img.shape: face_pred = face_pred.expand_as(test_img)
        if args.with_rec:
            face_pred_rec = face_pred.clamp(0, 255)
            face_rec = rec_model(face_pred_rec)
        comb_save_path = os.path.join(save_result_dir, 'comb_' + os.path.basename(test_img_path))
        sketch_save_path = os.path.join(save_result_dir, os.path.basename(test_img_path))
        save_img_list = []
        save_img_list.append(img_process.save_var_img(test_img, size=size))
        save_img_list.append(img_process.save_var_img(face_pred, size=size))
        if args.with_rec:
            save_img_list.append(img_process.save_var_img(face_rec, size=size))
        imgs_comb = np.hstack((np.array(i) for i in save_img_list))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(comb_save_path)
        save_img_list[1].save(sketch_save_path)
 
def test_rec(model, rec_model, args, save_weight_dir, save_weight_path):
    size = (250, 200) 
    model.eval()

    if not os.path.exists(args.result_root): os.mkdir(args.result_root)
    #  save_result_dir = args.result_root
    save_result_dir = os.path.join(args.result_root, 'result_{:02d}_rec'.format(args.test_epoch))
    if not os.path.exists(save_result_dir): os.mkdir(save_result_dir)
    
    model.load_state_dict(torch.load(os.path.join(save_weight_path, 'epochs-{:03d}-p2s.pth'.format(args.test_epoch))))
    if args.with_rec:
        rec_model.load_state_dict(torch.load(os.path.join(save_weight_path, 'epochs-{:03d}-rec.pth'.format(args.test_epoch))))

    for img_name in os.listdir(args.test_dir):
        test_img_path = os.path.join(args.test_dir, img_name)
        test_img = img_process.read_img_var(test_img_path, size=(256, 256))
        test_face_path = test_img_path.replace('test_sketches', 'test_photos')
        test_face = img_process.read_img_var(test_face_path)
        face_pred = model(test_img)
        if face_pred.shape != test_img.shape: face_pred = face_pred.expand_as(test_img)
        if args.with_rec:
            face_pred_rec = face_pred.clamp(0, 255)
            face_rec = rec_model(face_pred_rec)
        comb_save_path = os.path.join(save_result_dir, 'comb_' + os.path.basename(test_img_path))
        sketch_save_path = os.path.join(save_result_dir, os.path.basename(test_img_path))
        save_img_list = []
        save_img_list.append(img_process.save_var_img(test_img, size=size))
        save_img_list.append(img_process.save_var_img(face_pred, size=size))
        save_img_list.append(img_process.save_var_img(test_face, size=size))
        if args.with_rec:
            save_img_list.append(img_process.save_var_img(face_rec, size=size))
        imgs_comb = np.hstack((np.array(i) for i in save_img_list))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(comb_save_path)
        save_img_list[1].save(sketch_save_path)
   
if __name__ == '__main__':
    gm=GPUManager()
    torch.cuda.set_device(gm.auto_choice())
    args = cmd_option()
    #  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args.save_weight_dir = 'face2sketch-norm_G{}_D{}-top{}-flayers{}-weight-{:.1e}-{:.1e}-{:.1e}-epoch{:02d}-{}'.format(
                        args.Gnorm, args.Dnorm, args.topk, "".join(map(str, args.flayers)),
                        args.weight[0], args.weight[1], args.weight[2], 
                        args.epochs, args.other) 
    args.save_weight_path = os.path.join(args.weight_root, args.save_weight_dir)

    if args.train_eval == 'train':
        print('Saving weight path', args.save_weight_path)
        utils.mkdirs(args.save_weight_path)
        train(args)
    elif args.train_eval == 'eval':
        print('Loading weight path', save_weight_path)
        test(model, rec_model, args, save_weight_dir, save_weight_path)
    elif args.train_eval == 'eval_rec':
        print('Loading weight path', save_weight_path)
        test_rec(model, rec_model, args, save_weight_dir, save_weight_path)


