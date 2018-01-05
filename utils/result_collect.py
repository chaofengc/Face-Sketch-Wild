import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
from PIL import Image
import os

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s)]

def save_fig_pdf(img_lists, title, save_path, rows_per_page=5, page_size=(8.27, 11.69)):
    """
    save image list to pages of pdf files.

    """
    # The PDF document
    pdf_pages = PdfPages(save_path)

    grid_w = len(img_lists)
    grid_h = max([len(x) for x in img_lists])
    #  print(grid_h, grid_w)
    for row in range(grid_h):
        page_row = row % rows_per_page
        if not page_row:
            fig = plt.figure(figsize=page_size, dpi=100)
        for col in range(grid_w):
            ax = plt.subplot2grid((rows_per_page, grid_w), (page_row, col))
            if not page_row:
                ax.set_title(title[col])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(img_lists[col][row])
        if not (page_row + 1) % rows_per_page or row + 1 == grid_h:
            plt.tight_layout()
            pdf_pages.savefig(fig)
    pdf_pages.close()

def read_img_save_grid_pdf(path_lists, title, save_path):
    img_lists = []
    for d in path_lists:
        tmp_list = []
        for root, dirs, files in os.walk(d):
            for f in sorted(files):
                img = Image.open(os.path.join(root, f)).convert('RGB')
                tmp_list.append(img)
        img_lists.append(tmp_list)
    save_fig_pdf(img_lists, title, save_path)


if __name__ == '__main__':
    #  photo_dir = '../data/CUFS/test_photos'
    #  save_photo_dir = '../others_result/CUFS/Photo'
    #  CUHK_list = []
    #  AR_list = []
    #  XM_list = []
    #  for i in os.listdir(photo_dir):
        #  if 'CUHK' in i:
            #  CUHK_list.append(i)
        #  if 'AR' in i:
            #  AR_list.append(i)
        #  if 'XM' in i:
            #  XM_list.append(i)
    #  CUHK_list.sort(key=alphanum_key) 
    #  AR_list.sort(key=alphanum_key) 
    #  XM_list.sort(key=alphanum_key) 
    #  CUFS_list = CUHK_list + AR_list + XM_list 
    #  for idx, i in enumerate(CUFS_list):
        #  read_path = os.path.join(photo_dir, i)
        #  save_path = os.path.join(save_photo_dir, '{}.jpg'.format(idx+1))
        #  img = Image.open(read_path)
        #  img.save(save_path)

    methods = ['Photo', 'gt_sketch', 'MWF', 'SSD', 'FCN', 'GAN', 'BP-GAN', 'Fast-RSLCR', 'RSLCR']
    result_root = '../others_result/CUFS/'
    selected_images = np.random.choice(range(1, 339), 8, replace=False)
    comb_img = []
    for i in sorted(selected_images):
        tmp_img_row = []
        for m in methods:
            result_dir = os.path.join(result_root, m)
            img_path = os.path.join(result_dir, '{}.jpg'.format(i))
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            tmp_img_row.append(img_array)
        comb_img.append(np.hstack(tmp_img_row))
    comb_img = np.vstack(comb_img)
    comb_img = Image.fromarray(comb_img)
    comb_img.save('/home/cfchen/Dropbox/papers/face_sketch_IJCVSI/Image_test.png')
    exit()


    #  path_lists = [
    #  '../test/CUHK_student_test/photos',
    #  '../results/s2p-result-CUHK_student_test-model1-IN-pre0-tune60-weight-1.0e+00-0.0e+00-1.0e-04_largedata-vggmean-s2p/photo',
    #  '../results/p2s-CUHK_student_test-result-model1-IN-lr0.001-pre0-tune60-weight-1.0e+00-0.0e+00-1.0e-04_coloraug-largedata-vggmean/sketch',
    #  '../test/CUHK_student_test/sketches',
    #  ]
    #  title = ['GT Photo', 'Pred Photo', 'Pred Sketch', 'GT Sketch']
    #  save_pdf = '../result_fig/CUHK_student_test.pdf'
    #  read_img_save_grid_pdf(path_lists, title, save_pdf)

    #  path_lists = [
    #  '../test/AR_test/photos',
    #  '../results/s2p-result-AR_test-model1-IN-pre0-tune60-weight-1.0e+00-0.0e+00-1.0e-04_largedata-vggmean-s2p/photo',
    #  '../results/p2s-AR_test-result-model1-IN-lr0.001-pre0-tune60-weight-1.0e+00-0.0e+00-1.0e-04_coloraug-largedata-vggmean/sketch',
    #  '../test/AR_test/sketches',
    #  ]
    #  title = ['GT Photo', 'Pred Photo', 'Pred Sketch', 'GT Sketch']
    #  save_pdf = '../result_fig/AR_test.pdf'
    #  read_img_save_grid_pdf(path_lists, title, save_pdf)

    path_lists = [
    '../test/natural_face_test/photos',
    '../results/pix2pix-natural_face_test-AtoB-result-model1-IN-layers00111-loss_4-weight-1.0e+00-1.0e-01-1.0e-04_coloraug-sizeaug/result',
    '../results/sp2s-natural_face_test-AtoB-result-model1-IN-layers00111-loss_2-weight-0.0e+00-1.0e+00-1.0e-04_noaug/result',
    #  '../test/AR_test/sketches',
    ]
    title = ['Photo', 'Fully Supervise(1172 gt sketches)', 'Semi-Supervise (188 gt sketches)']
    save_pdf = '../result_fig/semi_supervise.pdf'
    read_img_save_grid_pdf(path_lists, title, save_pdf)



