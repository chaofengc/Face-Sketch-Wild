import os

#  gpus          = '1,2'
gpus           = '1'
batch_size     = 6
learning_rate  = 1e-3
epochs         = 40 
vgg_weight     = '/home/cfchen/pytorch_models/vgg_conv.pth'
weight_root    = '/disk1/cfchen/Face-Sketch-Wild/weight'
Gnorm          = 'IN'
Dnorm          = 'None'
feature_layers = [0, 0, 1, 1, 1]
weight         = [1e0, 1e3, 1e-5]  # style loss, adv loss, tv loss
resume         = 0
topk           = 5
vgg_select_num = 0
# ---- train style ['CUHK_student', 'AR', 'XM2VTS', 'CUFSF'] -------------
train_style = [1, 1, 1, 0]  #  train_style = [0, 0, 0, 1]  # train style ['CUHK_student', 'AR', 'XM2VTS', 'CUFSF']
#  other          = 'with_rec_clamp_sch-{}-vgg{:02d}-crop-ext'.format("".join(map(str, train_style)), vgg_select_num)
#  other          = 'with_rec{}_clamp_sch-{}-vgg{:02d}-semi'.format(with_rec, "".join(map(str, train_style)), vgg_select_num)
other          = 'cufs'
#  ref_feature  = './data/crop_feature_dataset.pth'
#  ref_img_list = './data/crop_dataset_img_list.txt'
ref_feature = './data/feature_dataset.pth'
ref_img_list = './data/dataset_img_list.txt'
train_data     = [
                 './data/AR/train_photos',
                 './data/CUHK_student/train_photos',
                 './data/XM2VTS/train_photos',
                 #  './data/CUFSF/train_photos',
                 './data/CUFSF_crop/train_photos',
                 #  './data/CUFSF_crop/test_photos',
                ] 
if vgg_select_num:
    train_data.append('./data/vggface_{:02d}/'.format(vgg_select_num))
param          = [
        '--gpus {}'.format(gpus),
        '--train-data {}'.format(" ".join(train_data)),
        '--train-style {}'.format(" ".join(map(str, train_style))),
        '--batch-size {}'.format(batch_size),
        '--epochs {}'.format(epochs),
        '--vgg19-weight {}'.format(vgg_weight),
        '--weight-root {}'.format(weight_root),
        '--Gnorm {}'.format(Gnorm),
        '--Dnorm {}'.format(Dnorm),
        '--weight {} {} {}'.format(*weight),
        '--flayers {} {} {} {} {}'.format(*feature_layers),
        '--topk {}'.format(topk),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        '--ref-feature {}'.format(ref_feature),
        '--ref-img-list {}'.format(ref_img_list),
        ]

os.system('python face2sketch_wild.py train {}'.format(" ".join(param)))
print(train_data, '\tdone, ')

