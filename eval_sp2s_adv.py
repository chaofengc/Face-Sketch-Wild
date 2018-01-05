from __future__ import print_function
import os
from quan_eval import avg_score
from beautifultable import BeautifulTable

disk           = '/disk1'
#  gpus          = '1,2'
gpus           = '2'
batch_size     = 6
learning_rate  = 1e-3
#  epochs         = 40
epochs         = 40
vgg_weight     = '/home/cfchen/pytorch_models/vgg_conv.pth'
weight_root    = os.path.join(disk, 'cfchen/e2e_facesketch/weight')
model_version  = 3
norm           = 'IN'
feature_layers = [0, 0, 1, 1, 1]
content_layers = [0, 0, 1, 0, 0]
weight         = [1e0, 1e0, 1e-3, 1e5]  # mse loss, feature loss, tv loss
with_rec       = 1
resume         = 0
topk           = 5
other          = 'with_rec_clamp_newmean-adv'
#  other          = 'with_rec_clamp_newmean-adv-1e4'
other          = 'with_rec_clamp_newmean-adv-5e4'
#  other          = 'with_rec_clamp_newmean-xm2vts'
model_param    = [
        '--gpus {}'.format(gpus),
        '--batch-size {}'.format(batch_size),
        '--lr {}'.format(learning_rate),
        '--epochs {}'.format(epochs),
        '--vgg19-weight {}'.format(vgg_weight),
        '--weight-root {}'.format(weight_root),
        '--model-version {}'.format(model_version),
        '--norm {}'.format(norm),
        '--weight {} {} {} {}'.format(*weight),
        '--flayers {} {} {} {} {}'.format(*feature_layers),
        '--clayers {} {} {} {} {}'.format(*content_layers),
        '--with-rec {}'.format(with_rec),
        '--topk {}'.format(topk),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        ]

datasets = ['CUHK_student', 'AR', 'XM2VTS', 'CUFS', 'CUFSF']
test_dir = './test/natural_face_test/photos' 
test_dir = os.path.join('./data', datasets[-2], 'test_photos')
#  test_epoch = epochs - 1 
table = BeautifulTable()
for test_epoch in range(10, epochs - 1):
#  for test_epoch in [26]:
    result_root = '/disk1/cfchen/e2e_facesketch/results/sp2s-{}-top{}-result-model{}-{}-flayers{}-clayers{}-weight-{:.1e}-{:.1e}-{:.1e}{}'.format(
                  test_dir.split('/')[-2], topk, model_version, norm, 
                  "".join(map(str, feature_layers)), "".join(map(str, content_layers)), 
                  weight[0], weight[1], weight[2], other)
    
    arguments = model_param + [
                '--test-dir {}'.format(test_dir),
                '--test-epoch {}'.format(test_epoch),
                '--result-root {}'.format(result_root)]
    os.system('python semi_sup_p2s_adv.py eval {}'.format(" ".join(arguments)))
    
    avg_ssim_score = avg_score(os.path.join(result_root, 'result_{:02d}'.format(test_epoch)), test_dir.replace('test_photos', 'test_sketches'), 'ssim')
    print('Average SSIM of {}: {}'.format(test_dir.split('/')[-2], avg_ssim_score))
    table.append_row([test_epoch, avg_ssim_score])
    print(table)



