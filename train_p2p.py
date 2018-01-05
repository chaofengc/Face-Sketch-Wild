import os

batch_size = 12
learning_rate = 1e-3
pre_iter = 1000
epochs = 100
vgg_weight = '/home/cfchen/pytorch_models/vgg_conv.pth'
model_version = 1
norm = 'IN'
weight = [1.0, 1e-2, 1e-4]
addxy = 0
DoG = 0
resume = 0
#  train_data = './small_data'
#  train_data = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/train'
#  train_data = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/cityscapes/train'
#  train_data = '/data2/cfchen/pytorch-CcleGAN-and-pix2pix/datasets/facades/train'
#  train_data = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/edges2handbags/train'
#  train_data = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/maps/train'
train_data = './face_sketch_data/CUHK_AR'
other = '_'
param = [
        '--train-data {}'.format(train_data),
        '--batch-size {}'.format(batch_size),
        '--lr {}'.format(learning_rate),
        #  '--pre-epochs {}'.format(pre_epochs),
        '--tune-epochs {}'.format(tune_epochs),
        '--vgg19-weight {}'.format(vgg_weight),
        '--model-version {}'.format(model_version),
        '--norm {}'.format(norm),
        '--weight {} {} {}'.format(*weight),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        ]

os.system('python photo_to_photo.py train {}'.format(" ".join(param)))
print(train_data, '\tdone')

