import os

gpus             = '2'
test_dir         = './data/CUFS/test_photos'
test_gt_dir      = './data/CUFS/test_sketches'
result_dir       = './result/CUFS'
test_weight_path = './weight/face2sketch-norm_GIN_DNone-top5-style_cufs-flayers00111-weight-1.0e+00-1.0e+03-1.0e-05-epoch40-vgg00/epochs-003-G.pth'
param            = [
        '--gpus {}'.format(gpus),
        '--test-dir {}'.format(test_dir),
        '--test-gt-dir {}'.format(test_gt_dir),
        '--result-dir {}'.format(result_dir),
        '--test-weight-path {}'.format(test_weight_path),
        ]

os.system('python face2sketch_wild.py eval {}'.format(" ".join(param)))




