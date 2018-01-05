from __future__ import print_function
import os
import numpy as np
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim, compare_psnr
from matplotlib import pyplot as plt
from beautifultable import BeautifulTable
import cv2 as cv

import shutil
import subprocess
import matlab_wrapper

def avg_score(test_dir, gt_dir, metric='ssim', smooth=True, sigma=75):
    all_score = []
    all_names = os.listdir(gt_dir)
    if metric in ['fsim', 'mdsi', 'gmsd', 'brisque']:
        matlab = matlab_wrapper.MatlabSession()

    for name in sorted(all_names):
        test_img = Image.open(os.path.join(test_dir, name)).convert('L')
        if smooth:
            test_img = cv.bilateralFilter(np.array(test_img),7,sigma,sigma)
        gt_img = Image.open(os.path.join(gt_dir, name)).convert('L')
        test_img = np.array(test_img)# / 255.0
        gt_img = np.array(gt_img)# / 255.0
        if metric in ['fsim', 'mdsi', 'gmsd', 'brisque']:
            matlab.put('imageRef', gt_img)
            matlab.put('imageDis', test_img)
        if metric == 'ssim':
            tmp_score = compare_ssim(gt_img, test_img, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        elif metric == 'psnr':
            tmp_score = compare_psnr(gt_img, test_img)
        elif metric == 'fsim':
            matlab.eval('[score, fsimc] = FeatureSIM(imageRef, imageDis)')
            tmp_score = matlab.get('score')
        elif metric == 'mdsi':
            gt_img = gt_img[..., np.newaxis].repeat(3, -1)
            test_img = test_img[..., np.newaxis].repeat(3, -1)
            matlab.put('imageRef', gt_img)
            matlab.put('imageDis', test_img)
            matlab.eval('score = MDSI(imageRef, imageDis)')
            tmp_score = matlab.get('score')
        elif metric == 'gmsd':
            matlab.eval('[score, quality_map] = GMSD(imageRef, imageDis)')
            tmp_score = matlab.get('score')
        elif metric == 'brisque':
            matlab.eval("addpath(genpath('./brisque'))")
            #  matlab.eval('score_ref = brisquescore(imageRef)')
            matlab.eval('score_dis = brisquescore(imageDis)')
            tmp_score = matlab.get('score_dis')
            
        all_score.append(tmp_score)
    return np.mean(np.array(all_score))

def comp_ssim(test_dir1, test_dir2, gt_dir, save_map_dir=None):
    ssim_score = 0.
    all_names = os.listdir(gt_dir)
    for name in all_names:
        test_img1 = Image.open(os.path.join(test_dir1, name)).convert('L')
        gt_img = Image.open(os.path.join(gt_dir, name)).convert('L')
        test_img2 = Image.open(os.path.join(test_dir2, name)).convert('L')
        test_img1 = np.array(test_img1)# / 255.0
        gt_img = np.array(gt_img)# / 255.0
        test_img2 = np.array(test_img2)
        tmp_score1, ssim_map1 = compare_ssim(gt_img, test_img1, full=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tmp_score2, ssim_map2 = compare_ssim(gt_img, test_img2, full=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        if tmp_score2 > tmp_score1 + 0.01:
            print(name, tmp_score1, tmp_score2)
            ssim_map1 = ssim_map1 * 255
            ssim_map2 = ssim_map2 * 255
            img_array = np.hstack((ssim_map1, ssim_map2))
            img = Image.fromarray(img_array.astype('uint8'))
            img.save(os.path.join(save_map_dir, name))

        #  print(name, tmp_score)

    return ssim_score / len(all_names)

def calculate_score(metric):
    testing_datasets = ['CUFS', 'CUFSF']
    testing_method = ['LLE','SSD','MRF','MWF','Fast-RSLCR','RSLCR','BP-GAN','GAN','FCN']

    print('Calculating {} score for {}'.format(metric, testing_method))
    result_dict = {}
    table = BeautifulTable()
    table.numeric_precision = 4
    table.column_headers = ["Method", "CUFS", "CUFS(smooth)", "CUFSF", "CUFSF(smooth)"]
    for i in testing_method:
        row = []
        row.append(i)
        for j in testing_datasets:
            gt_dir = os.path.join('./others_result', j, 'gt_sketch')
            test_dir = os.path.join('./others_result', j, i)
            row.append(avg_score(test_dir, gt_dir, metric, smooth=False))
            row.append(avg_score(test_dir, gt_dir, metric, smooth=True))
        print(row)
        result_dict[i] = row[1:]
        table.append_row(row)
    print(table)
    with open('./others_result/{}_score.txt'.format(metric), 'w+') as f:
        print(table, file=f)
    np.save('./others_result/{}_score.npy'.format(metric), result_dict)

def draw_bar(data_path, methods, metric):
    def autolabel(rects, ax, pos):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., pos*height,
                        '%.2f' % height, ha='center', va='bottom', fontsize=8)
    score = np.load(data_path).item()

    score_array = []
    for i in methods:
        score_array.append(score[i])
    score_array = np.array(score_array) * 100

    pos = np.arange(score_array.shape[0])
    bar_width = 0.4
    labels = ['Origin', 'Smooth']
    color = ['sandybrown', 'sienna']
    fig, ax = plt.subplots(2, 1)

    rect1 = ax[0].bar(pos, score_array[:, 0], bar_width, alpha=0.5, color=color[0], label=labels[0])
    rect2 = ax[0].bar(pos + bar_width, score_array[:, 1], bar_width, alpha=0.5, color=color[1], label=labels[1])
    autolabel(rect1, ax[0], 1.00)
    autolabel(rect2, ax[0], 1.02)
    ax[0].set_title('(a) {} Score on CUFS'.format(metric))
    ax[0].set_ylabel('{} Score (%)'.format(metric))
    if metric == 'SSIM': 
        ax[0].set_yticks(range(45, 60, 5))
        ax[0].set_ylim(45, 60)
    else:
        ax[0].set_yticks(range(60, 75, 5))
        ax[0].set_ylim(60, 75)
    ax[0].set_xticks(pos + 0.5*bar_width)
    ax[0].set_xticklabels(methods)
    ax[0].legend()
    ax[0].grid(linestyle='dotted')

    rect1 = ax[1].bar(pos, score_array[:, 2], bar_width, alpha=0.5, color=color[0], label=labels[0])
    rect2 = ax[1].bar(pos + bar_width, score_array[:, 3], bar_width, alpha=0.5, color=color[1], label=labels[1])
    autolabel(rect1, ax[1], 1.00)
    autolabel(rect2, ax[1], 1.02)
    ax[1].set_title('(b) {} Score on CUFSF'.format(metric))
    ax[1].set_ylabel('{} Score (%)'.format(metric))
    if metric == 'SSIM': 
        ax[1].set_yticks(range(35, 50, 5))
        ax[1].set_ylim(35, 50)
    else:
        ax[1].set_yticks(range(60, 75, 5))
        ax[1].set_ylim(60, 75)

    ax[1].set_xticks(pos + 0.5*bar_width)
    ax[1].set_xticklabels(methods)
    ax[1].legend()
    ax[1].grid(linestyle='dotted')

    plt.tight_layout()
    plt.savefig('/home/cfchen/Dropbox/papers/face_sketch_IJCVSI/Figure_{}_score.eps'.format(metric))

if __name__ == '__main__':
    #  calculate_score('ssim')
    #  calculate_score('fsim')
    calculate_score('brisque')

    method = ['LLE','SSD','MRF','MWF','RSLCR','BP-GAN','GAN','FCN']
    #  draw_bar('./others_result/ssim_score.npy', method, 'SSIM')
    #  draw_bar('./others_result/fsim_score.npy', method, 'FSIM')
    draw_bar('./others_result/brisque_score.npy', method, 'BRISQUE')
    # -----------  CUHK student ---------------------
    ours_dataset = ['AR', 'CUHK_student', 'XM2VTS', 'CUFS']
    dataset_idx = 3
    gt_dir = './data/{}/test_sketches'.format(ours_dataset[dataset_idx])
    ours_best = '/disk1/cfchen/e2e_facesketch/results/sp2s-{}-top5-result-model3-IN-flayers00111-clayers00100-weight-1.0e+00-1.0e+00-1.0e-03with_rec_clamp_sch/result_30/'.format(ours_dataset[dataset_idx])

    for i in ['brisque']:
        table = BeautifulTable()
        table.numeric_precision = 4
        table.column_headers = ["Method", "avg_{}_score".format(i)]
        table.append_row(['ours origin', avg_score(ours_best, gt_dir, i, smooth=False)])
        table.append_row(['ours smooth', avg_score(ours_best, gt_dir, i, smooth=True)])
        print(table)


    # ----------- CUFS
    #  gt_dir = './data/CUFS/test_sketches'
    #  ours_best = '/disk1/cfchen/e2e_facesketch/results/sp2s-CUFS-top5-result-model3-IN-flayers00111-clayers00100-weight-1.0e+00-1.0e+00-1.0e-03with_rec_clamp_newmean/result_25/'

    #  table = BeautifulTable()
    #  table.column_headers = ["Method", "avg_ssim_score"]
    #  table.append_row(['ours', avg_score(ours_best, gt_dir)])
    #  print(table)

