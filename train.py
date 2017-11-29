import os

gpuid = 0
batch_size = 16
learning_rate = 1e-3
pre_epochs = 0
tune_epochs = 60
vgg_weight = '/home/cfchen/pytorch_models/vgg_conv.pth'
addxy = 0
DoG = 0
model_version = 1
norm = 'IN'
weight = [1.0, 0, 1e-4]
addxy = 0
DoG = 0
resume = 1
train_data = './small_data'
train_data = './large_data'
other = '_coloraug-largedata-vggmean-shuffle'
param = [
        '--gpuid {}'.format(gpuid),
        '--train-data {}'.format(train_data),
        '--batch-size {}'.format(batch_size),
        '--lr {}'.format(learning_rate),
        '--pre-epochs {}'.format(pre_epochs),
        '--tune-epochs {}'.format(tune_epochs),
        '--vgg19-weight {}'.format(vgg_weight),
        '--addxy {}'.format(addxy),
        '--DoG {}'.format(DoG),
        '--model-version {}'.format(model_version),
        '--norm {}'.format(norm),
        '--weight {} {} {}'.format(*weight),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        ]

os.system('python photo_to_sketch.py train {}'.format(" ".join(param)))


