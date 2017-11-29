import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
from PIL import Image
import os

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
    path_lists = [
    '../test/CUHK_student_test/photos',
    '../results/s2p-result-CUHK_student_test-model1-IN-pre0-tune60-weight-1.0e+00-0.0e+00-1.0e-04_largedata-vggmean-s2p/photo',
    '../results/p2s-CUHK_student_test-result-model1-IN-lr0.001-pre0-tune60-weight-1.0e+00-0.0e+00-1.0e-04_coloraug-largedata-vggmean/sketch',
    '../test/CUHK_student_test/sketches',
    ]
    title = ['GT Photo', 'Pred Photo', 'Pred Sketch', 'GT Sketch']
    save_pdf = '../result_fig/CUHK_student_test.pdf'
    read_img_save_grid_pdf(path_lists, title, save_pdf)

    path_lists = [
    '../test/AR_test/photos',
    '../results/s2p-result-AR_test-model1-IN-pre0-tune60-weight-1.0e+00-0.0e+00-1.0e-04_largedata-vggmean-s2p/photo',
    '../results/p2s-AR_test-result-model1-IN-lr0.001-pre0-tune60-weight-1.0e+00-0.0e+00-1.0e-04_coloraug-largedata-vggmean/sketch',
    '../test/AR_test/sketches',
    ]
    title = ['GT Photo', 'Pred Photo', 'Pred Sketch', 'GT Sketch']
    save_pdf = '../result_fig/AR_test.pdf'
    read_img_save_grid_pdf(path_lists, title, save_pdf)




