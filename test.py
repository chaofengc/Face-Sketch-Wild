import os
import sys

gpus             = '2'
if sys.argv[1] == '1':
    test_dir         = './data/CUFS/test_photos'
    test_gt_dir      = './data/CUFS/test_sketches'
    result_dir       = './result/CUFS'
    test_weight_path = './pretrain_model/cufs-epochs-021-meanshift30-G.pth'
elif sys.argv[1] == '2':
    test_dir         = './data/CUFSF/test_photos'
    test_gt_dir      = './data/CUFSF/test_sketches'
    result_dir       = './result/CUFSF'
    test_weight_path = './pretrain_model/cufsf-epochs-021-G.pth'
elif sys.argv[1] == '3':
    test_dir         = './data/CUHK_student/test_photos'
    test_gt_dir      = './data/CUHK_student/test_sketches'
    result_dir       = './result/CUHK_student'
    test_weight_path = './pretrain_model/cufs-epochs-021-meanshift30-G.pth'
elif sys.argv[1] == '4':
    test_dir         = './data/vgg_test/'
    test_gt_dir      = 'none' 
    result_dir       = './result/VGG'
    test_weight_path = './pretrain_model/vgg-epochs-002-G.pth'

param            = [
        '--gpus {}'.format(gpus),
        '--test-dir {}'.format(test_dir),
        '--test-gt-dir {}'.format(test_gt_dir),
        '--result-dir {}'.format(result_dir),
        '--test-weight-path {}'.format(test_weight_path),
        ]

os.system('python face2sketch_wild.py eval {}'.format(" ".join(param)))




