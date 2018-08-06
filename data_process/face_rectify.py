"""
Rectify the face photo according to the paper: Real-Time Exemplar-Based Face Sketch Synthesis. 
    shape:      h=250, w=200
    position:   left eye (x=75,y=125), right eye (x=125, y=125)

This module use similarity transformation to roughly align the two eyes.
Specifically, the transformation matrix can be written as:
    S = |s_x cos(\theta), sin(\theta)    , t_x | 
        |-sin(\theta)   , s_y cos(\theta), t_y |
There are 5 degrees in the above function, needs at least 3 points(x, y) to solve it.
we can simply hallucinate a third point such that it forms an equilateral triangle with the two known points.

Reference: 
    http://www.learnopencv.com/average-face-opencv-c-python-tutorial/
    http://blog.csdn.net/GraceDD/article/details/51382952
"""
import math
import numpy as np
import os

import dlib
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted

def detect_fiducial_points(img, predictor_path):
    """
    Detect face landmarks and return the mean points of left and right eyes.
    If there are multiple faces in one image, only select the first one.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    dets = detector(img, 1)
    if len(dets) < 1:
        return [] 
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        break
    landmarks = []
    for i in range(68):
        landmarks.append([shape.part(i).x, shape.part(i).y])
    landmarks = np.array(landmarks)
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]
    return np.array([np.mean(left_eye, 0), np.mean(right_eye, 0)]).astype('int')


def similarityTransform(inPoints, outPoints) :
    """
    Calculate similarity transform:
    Input:
        (left eye, right eye) in (x, y)
        inPoints: (2, 2), numpy array. 
        outPoints: (2, 2), numpy array
    Return:
        A partial affine transform.
    """
    s60 = math.sin(60*math.pi/180) 
    c60 = math.cos(60*math.pi/180) 

    inPts  = np.copy(inPoints).tolist() 
    outPts = np.copy(outPoints).tolist() 
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0] 
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1] 
    inPts.append([np.int(xin), np.int(yin)]) 

    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0] 
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1] 
    outPts.append([np.int(xout), np.int(yout)]) 
    tform = cv.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False) 

    return tform

def rectify_img(img_path, predictor_path):
    template_eye_pos = np.array([[75, 125], [125, 125]])
    template_size = (200, 250)
    img = cv.imread(img_path) 
    detected_eyes = detect_fiducial_points(np.array(img), predictor_path)
    if not len(detected_eyes):
        return None 
    trans = similarityTransform(detected_eyes, template_eye_pos) 
    rect_img = cv.warpAffine(img, trans, template_size)
    return rect_img

def align_img(ref_path, src_path, predictor_path):
    ref_img = cv.imread(ref_path)
    src_img = cv.imread(src_path)

    ref_eyes = detect_fiducial_points(np.array(ref_img), predictor_path)
    src_eyes = detect_fiducial_points(np.array(src_img), predictor_path)
    trans = similarityTransform(src_eyes, ref_eyes)
    rect_img = cv.warpAffine(src_img, trans, (200, 250))
    return rect_img


if __name__ == '__main__':
    src_dir = '../result_ours/CUFSF_intersect/ours_result'
    ref_dir = '../result_ours/CUFSF_intersect/gt_sketch'

    save_dir = '../result_ours/CUFSF_intersect/ours_warp'
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    ref_img_list = natsorted(os.listdir(ref_dir))
    src_img_list = natsorted(os.listdir(src_dir))

    for i in range(len(ref_img_list)):
        ref_path = os.path.join(ref_dir, ref_img_list[i])
        src_path = os.path.join(src_dir, src_img_list[i])
        save_path = os.path.join(save_dir, ref_img_list[i])
        warp_src = align_img(ref_path, src_path, './shape_predictor_68_face_landmarks.dat') 
        cv.imwrite(save_path, warp_src)
    #  template_eye_pos = np.array([[75, 125], [125, 125]])
    #  template_size = (200, 250)
    #  img_path = '/disk1/cfchen/data/FERET/original_photo/00001.jpg'
    #  img = cv.imread(img_path) 
    #  detected_eyes = detect_fiducial_points(np.array(img), './shape_predictor_68_face_landmarks.dat')
    #  trans = similarityTransform(detected_eyes, template_eye_pos) 
    #  rect_img = cv.warpAffine(img, trans, template_size)
    #  cv.imshow('test', rect_img)
    #  cv.waitKey()
    


