import os
import bz2

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import dlib
import openface

data_root_dir = '/disk2/cfchen_data/FERET/colorferet/colorferet/'
img_name_file = open('/disk2/cfchen_data/FERET/feret_filenames.txt')

img_names = [line.strip() for line in img_name_file.readlines()] 

def write_img():
    for name in img_names:
        name_fields = name[:-4].split('_')
        key1 = name_fields[0][:5]
        key2 = name_fields[1]
        key3 = name_fields[0][5:7]
        
        if int(name_fields[0][:5]) < 740:
            photo_dir = os.path.join(data_root_dir, 'dvd1/data/images', name_fields[0][:5]) 
        else:
            photo_dir = os.path.join(data_root_dir, 'dvd2/data/images', name_fields[0][:5]) 
    
        if not os.path.exists(photo_dir):
            continue
    
        for root, dirs, files in os.walk(photo_dir):
            for f in files:
                if key1 in f and key2 in f and key3 in f:
                    break
    
        photo_full_path = os.path.join(photo_dir, f)
        if photo_full_path.endswith('bz2'):
            os.system('bzip2 -d {}'.format(photo_full_path))
            photo_full_path = photo_full_path[:-4]
    
        img = cv.imread(photo_full_path)
        save_path = '/disk2/cfchen_data/FERET/original_photo/{}.jpg'.format(key1)
        print 'Writing image', save_path
        cv.imwrite(save_path, img)

def test_img(photo_dir, sketch_dir):

    for root, dirs, files in os.walk(photo_dir):
        for f in files[::10]:
            photo = cv.imread(os.path.join(photo_dir, f))
            photo = photo[..., [2, 1, 0]]
            sketch = cv.imread(os.path.join(sketch_dir, f))
            fig = plt.figure('test')
            fig.add_subplot(121)
            plt.imshow(photo)
            fig.add_subplot(122)
            plt.imshow(sketch)
            plt.waitforbuttonpress()

def detect_fiducial_points(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    dets = detector(img, 1)
    if len(dets) < 1:
        return [] 
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        #  print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
    landmarks = []
    for i in range(68):
        landmarks.append([shape.part(i).x, shape.part(i).y])
    landmarks = np.array(landmarks)
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]
    return np.array([np.mean(left_eye, 0), np.mean(right_eye, 0), np.mean(mouth, 0)]).astype('float32')
    

def crop(points, img):
    img_h, img_w = img.shape[:2]
    scale = 3.8

    wh_ratio = 200. / 250.
    center = np.mean(points, 0)
    width = min((points[1, 0] - points[0, 0]) * scale, img_w)
    height = min(width / wh_ratio, img_h)
    size = np.array([width, height])
    lt = (center - size / 2.).astype('int')
    rb = (center + size / 2.).astype('int')
    lt = np.clip(lt, [0, 0], [img_w, img_h]).astype('int')
    rb = np.clip(rb, [0, 0], [img_w, img_h]).astype('int')
    crop_img = img[lt[1]:rb[1], lt[0]:rb[0]]
    crop_img = cv.resize(crop_img, (200, 250))
    return crop_img

def crop_img(photo_dir, sketch_dir, save_photo_dir, save_sketch_dir):
    for name in img_names:
        name = os.path.splitext(name)[0]
        #  face_points_file = open(os.path.join('/disk2/cfchen_data/FERET/photo_points', name + '.3pts'))
        #  sketch_points_file = open(os.path.join('/disk2/cfchen_data/FERET/sketch_points', name[:5] + '.3pts'))
        #  face_points = [l.strip() for l in face_points_file.readlines()]
        #  sketch_points = [l.strip() for l in sketch_points_file.readlines()]
        #  face_points = np.array([[int(float(y)) for y in x.split(' ')] for x in face_points])
        #  sketch_points = np.array([[int(float(y)) for y in x.split(' ')] for x in sketch_points]) 

        photo_img_path = os.path.join(photo_dir, name[:5] + '.jpg')
        sketch_img_path = os.path.join(sketch_dir, name[:5] + '.jpg')
        if os.path.exists(photo_img_path):
            photo_img = cv.imread(photo_img_path, 1)
            sketch_img = cv.imread(sketch_img_path, 0)

            #  photo_img = cv.copyMakeBorder(photo_img, 0, 0, 100, 100, cv.BORDER_CONSTANT, 0)
            face_points = detect_fiducial_points(photo_img)
            sketch_points = detect_fiducial_points(sketch_img)

            if not len(face_points) or not len(sketch_points):
                continue
            
            H = cv.getAffineTransform(sketch_points, face_points)
            sketch_img = cv.warpAffine(sketch_img, H, (photo_img.shape[1], photo_img.shape[0]))

            sketch_points = detect_fiducial_points(sketch_img)
            if not len(sketch_points):
                continue

            crop_photo = crop(face_points, photo_img)
            crop_sketch = crop(sketch_points, sketch_img)
            cv.imwrite(os.path.join(save_photo_dir, name + '.jpg'), crop_photo)
            cv.imwrite(os.path.join(save_sketch_dir, name + '.jpg'), crop_sketch)
            print name + '.jpg sketch and photo saved'
            continue

            fig = plt.figure('test')
            ax = fig.add_subplot(221)
            plt.imshow(photo_img[...,[2, 1, 0]])
            ax.plot(face_points[:, 0], face_points[:, 1], 'ro')
            ax = fig.add_subplot(222)
            plt.imshow(sketch_img, cmap='gray')
            ax.plot(sketch_points[:, 0], sketch_points[:, 1], 'ro')
            crop_photo = crop(face_points, photo_img)
            crop_sketch = crop(sketch_points, sketch_img)
            fig.add_subplot(223)
            plt.imshow(crop_photo[..., [2, 1, 0]])
            fig.add_subplot(224)
            plt.imshow(crop_sketch, cmap='gray')
            plt.waitforbuttonpress()

        
if __name__ == '__main__':
    #  write_img()
    photo_dir = '/disk2/cfchen_data/FERET/original_photo'
    sketch_dir = '/disk2/cfchen_data/FERET/original_sketch'
    save_photo_dir = '/disk2/cfchen_data/FERET/crop_photo'
    save_sketch_dir = '/disk2/cfchen_data/FERET/crop_sketch'
    #  test_img(photo_dir, sketch_dir)
    crop_img(photo_dir, sketch_dir, save_photo_dir, save_sketch_dir)


    
