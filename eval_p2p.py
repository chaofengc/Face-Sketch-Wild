import os

data_root = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes'
#  data_root = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/cityscapes'
#  data_root = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/facades'
#  data_root = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/edges2handbags'
#  data_root = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/maps'
#  data_root = '/data2/cfchen/pytorch-CycleGAN-and-pix2pix/datasets/cityscapes'

learning_rate = 1e-3
pre_epochs = 5
tune_epochs = 60
vgg_weight = '/home/cfchen/pytorch_models/vgg_conv.pth'
model_version = 1
norm = 'IN'
weight = [1.0, 1.0, 1e-4]
addxy = 0
DoG = 0
resume = 0
train_data = os.path.join(data_root, 'train') 
other = '_'
model_param = [
        '--train-data {}'.format(train_data),
        '--lr {}'.format(learning_rate),
        '--pre-epochs {}'.format(pre_epochs),
        '--tune-epochs {}'.format(tune_epochs),
        '--vgg19-weight {}'.format(vgg_weight),
        '--model-version {}'.format(model_version),
        '--norm {}'.format(norm),
        '--weight {} {} {}'.format(*weight),
        '--other {}'.format(other),
        '--resume {}'.format(resume),
        ]

test_dir = os.path.join(data_root, 'val') 
test_pre_model = 'pretrain-{:03d}.pth'.format(pre_epochs - 1) 
test_tune_model = 'tune-{:03d}.pth'.format(5)
result_root = './results/pix2pix-{}-result-model{}-{}-pre{}-tune{}-weight-{:.1e}-{:.1e}-{:.1e}{}'.format(
              data_root.split('/')[-1], model_version, norm, pre_epochs, tune_epochs,
              weight[0], weight[1], weight[2], other)


for root, dirs, names in os.walk(test_dir):
    for n in sorted(names[:10]):
        arguments = model_param + [
                    '--test-img ' + os.path.join(root, n),
                    '--test-pre-model {}'.format(test_pre_model),
                    '--test-tune-model {}'.format(test_tune_model),
                    '--result-root {}'.format(result_root)]
        #  print arguments

        os.system('python photo_to_photo.py eval {}'.format(" ".join(arguments)))

