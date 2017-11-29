import os

gpuid = 0
pre_epochs = 0
tune_epochs = 60 
learning_rate = 1e-3
addxy = 0
DoG = 0
model_version = 1
norm = 'IN'
weight = [1.0, 0, 1e-4]
addxy = 0
DoG = 0
other = '_coloraug-largedata-vggmean'
model_param = [
        '--pre-epochs {}'.format(pre_epochs),
        '--tune-epochs {}'.format(tune_epochs),
        '--addxy {}'.format(addxy),
        '--DoG {}'.format(DoG),
        '--model-version {}'.format(model_version),
        '--norm {}'.format(norm),
        '--weight {} {} {}'.format(*weight),
        '--other {}'.format(other),
        '--lr {}'.format(learning_rate),
        ]

test_dir = './test'
test_dir = './test/AR_test/photos'
test_dir = './test/CUHK_student_test/photos'
test_pre_model = 'pretrain-{:03d}.pth'.format(pre_epochs - 1) 
test_tune_model = 'tune-{:03d}.pth'.format(59)
result_root = './results/p2s-{}-result-model{}-{}-lr{}-pre{}-tune{}-weight-{:.1e}-{:.1e}-{:.1e}{}'.format(
              test_dir.split('/')[-2], model_version, norm, learning_rate, pre_epochs, tune_epochs,
              weight[0], weight[1], weight[2], other)

for root, dirs, names in os.walk(test_dir):
    for n in sorted(names):
        arguments = model_param + ['--gpuid {}'.format(gpuid),
                    '--test-img ' + os.path.join(root, n),
                    '--test-pre-model {}'.format(test_pre_model),
                    '--test-tune-model {}'.format(test_tune_model),
                    '--result-root {}'.format(result_root)]
        #  print arguments

        os.system('python photo_to_sketch.py eval {}'.format(" ".join(arguments)))

