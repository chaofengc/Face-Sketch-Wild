import os

gpus = '1,2'
batch_size = 32
learning_rate = 1e-3
epochs = 100
vgg_weight = '/home/cfchen/pytorch_models/vgg_conv.pth'
weight_root = '/disk1/cfchen/e2e_facesketch/weight'
model_version = 1
norm = 'IN'
layers = [0, 0, 1, 0, 0]
weight = [0e0, 1e0, 1e-4]  # mse loss, feature loss, tv loss
loss_func = 2
direction = "AtoB"
#  direction = "BtoA"
resume = 0
time_steps = 1
other = '_sizeaug-coloraug-featuremse'
train_data = './face_sketch_data/CUHK_AR'
param = [
        '--gpus {}'.format(gpus),
        '--train-data {}'.format(train_data),
        '--batch-size {}'.format(batch_size),
        '--lr {}'.format(learning_rate),
        '--epochs {}'.format(epochs),
        '--vgg19-weight {}'.format(vgg_weight),
        '--weight-root {}'.format(weight_root),
        '--model-version {}'.format(model_version),
        '--norm {}'.format(norm),
        '--weight {} {} {}'.format(*weight),
        '--layers {} {} {} {} {}'.format(*layers),
        '--loss-func {}'.format(loss_func),
        '--direction {}'.format(direction),
        '--time-steps {}'.format(time_steps),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        ]

os.system('python photo_to_sketch.py train {}'.format(" ".join(param)))
print(train_data, '\tdone, ')

