import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import random

face_dir = '/disk1/cfchen/data/FERET/crop_photo'
sketch_dir = '/disk1/cfchen/data/FERET/crop_sketch'

for root, dirs, files in os.walk(face_dir):
    file_list = files
random.shuffle(file_list)

for idx, item in enumerate(file_list):
    if idx < 250:
        shutil.copy2(os.path.join(face_dir, item), './data/CUFSF/test_photos')
        shutil.copy2(os.path.join(sketch_dir, item), './data/CUFSF/test_sketches')
    else:
        shutil.copy2(os.path.join(face_dir, item), './data/CUFSF/train_photos')
        shutil.copy2(os.path.join(sketch_dir, item), './data/CUFSF/train_sketches')
