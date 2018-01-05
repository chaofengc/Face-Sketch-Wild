import os

# Modification area
disk           = '/disk1'
#  gpus          = '1,2'
gpus           = '1'
batch_size     = 6
learning_rate  = 1e-3
epochs         = 40
vgg_weight     = '/home/cfchen/pytorch_models/vgg_conv.pth'
weight_root    = os.path.join(disk, 'cfchen/e2e_facesketch/weight')
model_version  = 3
norm           = 'IN'
feature_layers = [0, 0, 1, 1, 1]
content_layers = [0, 0, 1, 0, 0]
weight         = [1e0, 1e0, 1e-3, 5e4]  # mse loss, feature loss, tv loss
with_rec       = 1
resume         = 0
topk           = 5
vgg_select_num = 1
other          = 'with_rec_clamp_newmean-ada'
other          = 'with_rec_clamp_newmean-sch'
other          = 'with_rec_clamp_newmean-adv-5e4'
other          = 'with_rec_clamp_sch-cufsf-vgg{:02d}'.format(vgg_select_num)
#  other          = 'with_rec_clamp_sch-cufsf'
direction      = "AtoB"
train_data     = [
                 './data/AR/train_photos',
                 './data/CUHK_student/train_photos',
                 './data/XM2VTS/train_photos',
                 './data/CUFSF/train_photos',
                 './data/vggface_{:02d}/train_photos'.format(vgg_select_num),
                ] 
param          = [
        '--gpus {}'.format(gpus),
        '--train-data {}'.format(" ".join(train_data)),
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
        '--direction {}'.format(direction),
        '--topk {}'.format(topk),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        ]

os.system('python semi_sup_p2s_adv.py train {}'.format(" ".join(param)))
print(train_data, '\tdone, ')

