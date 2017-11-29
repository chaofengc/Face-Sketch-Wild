import os

batch_size = 16
learning_rate = 1e-3
epochs = 100
vgg_weight = '/home/cfchen/pytorch_models/vgg_conv.pth'
model_version = 1
norm = 'IN'
layers = [0, 0, 1, 1, 1]
weight = [1e0, 1e0, 1e-4]  # mse loss, feature loss, tv loss
loss_func = 0
#  direction = "AtoB"
direction = "BtoA"
resume = 0
other = '_sizeaug'
#  train_data = './small_data'
#  train_data = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/train'
#  train_data = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/cityscapes/train'
#  train_data = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/facades/train'
#  train_data = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/edges2handbags/train'
#  train_data = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/maps/train'
train_data = './face_sketch_data/CUHK_AR'
param = [
        '--train-data {}'.format(train_data),
        '--batch-size {}'.format(batch_size),
        '--lr {}'.format(learning_rate),
        '--epochs {}'.format(epochs),
        '--vgg19-weight {}'.format(vgg_weight),
        '--model-version {}'.format(model_version),
        '--norm {}'.format(norm),
        '--weight {} {} {}'.format(*weight),
        '--layers {} {} {} {} {}'.format(*layers),
        '--loss-func {}'.format(loss_func),
        '--direction {}'.format(direction),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        ]

os.system('python photo_to_photo_multiscale.py train {}'.format(" ".join(param)))
print(train_data, '\tdone, ')

