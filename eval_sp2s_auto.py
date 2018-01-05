import os

# Modification area
disk           = '/disk1'
#  gpus          = '1,2'
gpus           = '1'
batch_size     = 12 
learning_rate  = 1e-3
epochs         = 40
vgg_weight     = '/home/cfchen/pytorch_models/vgg_conv.pth'
weight_root    = os.path.join(disk, 'cfchen/e2e_facesketch/weight')
model_version  = 3
norm           = 'IN'
#  feature_layers = [0, 0, 1, 1, 1]
feature_layers = [0, 0, 0, 0, 1]
content_layers = [0, 0, 1, 0, 0]
content_layers = [1, 0, 0, 0, 0]
weight         = [1e0, 0e0, 0e-5]  # content loss, feature loss, tv loss
with_rec       = 0  # 1: mse-rc; 0: feat-mse; -1: mse; 
hshift = 0
resume         = 0
topk           = 1
#  other          = 'with_rec_clamp_newmean-ada'
vgg_select_num = 0
train_style = [1, 1, 0, 0]  # train style ['CUHK_student', 'AR', 'XM2VTS', 'CUFSF']
other = 'autoencoder-{}-{}'.format(with_rec, hshift)
direction      = "AtoB"
train_data     = [
                 './data/AR/train_sketches',
                 './data/CUHK_student/train_sketches',
                 #  './data/XM2VTS/train_sketches',
                 #  './data/CUFSF/train_sketches',
                ] 
if vgg_select_num:
    train_data.append('./data/vggface_{:02d}/train_photos'.format(vgg_select_num))
model_param          = [
        '--gpus {}'.format(gpus),
        '--train-data {}'.format(" ".join(train_data)),
        '--train-style {}'.format(" ".join(map(str, train_style))),
        '--batch-size {}'.format(batch_size),
        '--lr {}'.format(learning_rate),
        '--epochs {}'.format(epochs),
        '--vgg19-weight {}'.format(vgg_weight),
        '--weight-root {}'.format(weight_root),
        '--model-version {}'.format(model_version),
        '--norm {}'.format(norm),
        '--weight {} {} {}'.format(*weight),
        '--flayers {} {} {} {} {}'.format(*feature_layers),
        '--clayers {} {} {} {} {}'.format(*content_layers),
        '--with-rec {}'.format(with_rec),
        '--hshift {}'.format(hshift),
        '--direction {}'.format(direction),
        '--topk {}'.format(topk),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        ]

test_dir = './data/CUHK_student/test_sketches'
test_epoch = epochs - 1
result_root = '/disk1/cfchen/e2e_facesketch/results/sp2s-{}-top{}-result-model{}-{}-flayers{}-clayers{}-weight-{:.1e}-{:.1e}-{:.1e}{}'.format(
                  test_dir.split('/')[-2], topk, model_version, norm, 
                  "".join(map(str, feature_layers)), "".join(map(str, content_layers)), 
                  weight[0], weight[1], weight[2], other)

arguments = model_param + [
            '--test-dir {}'.format(test_dir),
            '--test-epoch {}'.format(test_epoch),
            '--result-root {}'.format(result_root)]
os.system('python semi_sup_p2s_auto.py eval {}'.format(" ".join(arguments)))



