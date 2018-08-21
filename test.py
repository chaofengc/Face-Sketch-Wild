import os
import sys

gpus             = '2'
if sys.argv[1] == '1':
    test_dir         = './data/CUFS/test_photos'
    test_gt_dir      = './data/CUFS/test_sketches'
    result_dir       = './result/CUFS'
    test_weight_path = './pretrain_model/cufs-epochs-020-G.pth'
elif sys.argv[1] == '2':
    test_dir         = './data/CUFSF_crop/test_photos'
    test_gt_dir      = './data/CUFSF_crop/test_sketches'
    result_dir       = './result/CUFSF_crop'
    test_weight_path = './pretrain_model/cufsf-epochs-021-G.pth'

param            = [
        '--gpus {}'.format(gpus),
        '--test-dir {}'.format(test_dir),
        '--test-gt-dir {}'.format(test_gt_dir),
        '--result-dir {}'.format(result_dir),
        '--test-weight-path {}'.format(test_weight_path),
        ]

os.system('python face2sketch_wild.py eval {}'.format(" ".join(param)))




