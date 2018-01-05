from __future__ import print_function
import os

gpus = '1'
epochs = 500
learning_rate = 1e-4
weight_root = '/disk1/cfchen/e2e_facesketch/weight'
vgg_weight = '/home/cfchen/pytorch_models/vgg_conv.pth'
model_version = 1
norm = 'IN'
layers = [0, 0, 1, 0, 0]
#  layers = [0, 0, 0, 0, 1]
weight = [0e0, 1e0, 1e-4]  # mse loss, feature loss, tv loss
loss_func = 2
direction = "AtoB"
#  direction = "BtoA"
resume = 0
#  other = '_coloraug-sizeaug'
other = '_sizeaug-coloraug-featuremse'
train_data = './face_sketch_data/CUHK_AR'
model_param = [
        '--gpus {}'.format(gpus),
        '--train-data {}'.format(train_data),
        '--epochs {}'.format(epochs),
        '--lr {}'.format(learning_rate),
        '--vgg19-weight {}'.format(vgg_weight),
        '--weight-root {}'.format(weight_root),
        '--model-version {}'.format(model_version),
        '--norm {}'.format(norm),
        '--weight {} {} {}'.format(*weight),
        '--loss-func {}'.format(loss_func),
        '--layers {} {} {} {} {}'.format(*layers),
        '--direction {}'.format(direction),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        ]

test_dir = './test/natural_face_test/photos' 
#  test_model = 'epochs-{:03d}.pth'.format(epochs - 1) 
test_model = 'epochs-{:03d}.pth'.format(460) 
result_root = './results/pix2pix-{}-{}-result-model{}-{}-layers{}-loss_{}-weight-{:.1e}-{:.1e}-{:.1e}{}'.format(
              test_dir.split('/')[-2], direction, model_version, norm, 
              "".join(map(str, layers)), loss_func, weight[0], weight[1], weight[2], other)


for root, dirs, names in os.walk(test_dir):
    for n in sorted(names):
        arguments = model_param + [
                    '--test-img ' + os.path.join(root, n),
                    '--test-model {}'.format(test_model),
                    '--result-root {}'.format(result_root)]
        os.system('python photo_to_sketch.py eval {}'.format(" ".join(arguments)))
        #  break


