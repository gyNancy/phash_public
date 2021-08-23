from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import random
import tarfile
import scipy.misc

import numpy as np
from six.moves import urllib
import tensorflow as tf
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import imagehash
import robusthash
import math
import cv2
global_threshold= 0
def gen_image(arr):
    # two_d = (np.reshape(arr, (28, 28)) * 255 ).astype(np.uint8)
    #
    # img = Image.fromarray(two_d)
    fig = np.around((arr+0.5) * 255.0)
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)

    return img

def pfunction(arr, arr1):
    a = []
    differences = []

    for i in range(arr.shape[0]):

        a.append(max(0, 1-  ((imagehash.phash(gen_image(arr[i])) -  imagehash.phash(gen_image(arr1)) ) / 8 )  ))
    #     differences.append(imagehash.phash(gen_image(arr[i])) - imagehash.phash(gen_image(arr1)))
    # print('how is it possible ', differences)
    a = np.asarray(a)
    a = a.astype('float32')

    return a


def pfunction_log(arr, arr1):
    a = []
    for i in range(arr.shape[0]):
        a.append(max(0, -math.log((imagehash.phash(gen_image(arr[i])) -  imagehash.phash(gen_image(arr1)) ) + 1e-30) ))

    a = np.asarray(a)
    a = a.astype(float)
    return a
def pfunction_square(arr, arr1):
    a = []
    for i in range(arr.shape[0]):
        a.append((max(0, 1 - ((imagehash.phash(gen_image(arr[i])) - imagehash.phash(gen_image(arr1))) / 8)))**2)

    a = np.asarray(a)
    a = a.astype('float32')
    return a

def pfunction_tanh(arr, arr1):
    a = []
    for i in range(arr.shape[0]):
        a.append(max(0, np.tanh(1 - ((imagehash.phash(gen_image(arr[i]), global_bits, global_factor) - imagehash.phash(gen_image(arr1), global_bits, global_factor)) /global_threshold))))

    a = np.asarray(a)
    a = a.astype('float32')
    return a

# this function is used for robust hash
def pfunction_tanh2(arr, arr1):
    a = []
    timage = gen_image(arr1)
    if timage.mode == '1' or timage.mode == 'L' or timage.mode == 'P':
        timage = timage.convert('RGB')
    for i in range(arr.shape[0]):
        newimage = gen_image(arr[i])
        if newimage.mode == '1' or newimage.mode == 'L' or newimage.mode == 'P':
            newimage = newimage.convert('RGB')
        #a.append(max(0, np.tanh(1 - ((imagehash.average_hash(gen_image(arr[i])) - imagehash.average_hash(gen_image(arr1))) /global_threshold))))
        a.append(max(0, np.tanh( 1 - sum(1 for i, j in zip(robusthash.blockhash(newimage), robusthash.blockhash(timage)) if i != j)/ global_threshold )))
    a = np.asarray(a)
    a = a.astype('float32')
    return a

def pfunction_tanh3(arr, arr1):
    a = []
    for i in range(arr.shape[0]):
        a.append(max(0, np.tanh(1 - ((imagehash.dhash(gen_image(arr[i])) - imagehash.dhash(gen_image(arr1))) /global_threshold))))

    a = np.asarray(a)
    a = a.astype('float32')
    return a

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def pfunction_sigmoid(arr, arr1):
    a = []
    differences = []
    for i in range(arr.shape[0]):
        a.append(max(0, sigmoid(1 - ((imagehash.phash(gen_image(arr[i])) - imagehash.phash(gen_image(arr1))) / 8)) -0.5))
        differences.append(imagehash.phash(gen_image(arr[i])) - imagehash.phash(gen_image(arr1)))
    print('how is it possible ', differences)
    a = np.asarray(a)
    a = a.astype('float32')
    return a

# def pfunction_sift(arr):
#     sift = cv2.SIFT_create()
#     sum = []
#     for i in range(arr.shape[0]):
        

#         kp, des = sift.detectAndCompute(newimg_grey,None)
#         each_sum = 0
#         for i in range(len(des) - 2):
#             if np.linalg.norm(des[i] - des[i+1]) / np.linalg.norm(des[i] - des[i+2]) < 0.6:
#                 each_sum += np.linalg.norm(des[i] - des[i+2]) / np.linalg.norm(des[i] - des[i+1])
#         sum.append(each_sum)    
#     sum = np.array(sum)    
#     print('sum shape', sum.shape)
#     return sum   

def readimg(ff):
  f = "./real_fake/faces/"+ff
  #f = "./ImageNet_sorted/"+ff
  img = Image.open(f)
  img_gray = img.convert("L")
  img = np.array(img)
  gray_img = np.array(img_gray)
  img = resize(img,(224,224, 3), anti_aliasing=True)
  gray_img = resize(gray_img,(224,224, 1), anti_aliasing=True)

  gray_img = gray_img - 0.5
  img = img - 0.5
  if img.shape != (224,224, 3):
    return None
  return [img, gray_img]


class Face:
  def __init__(self):
    from multiprocessing import Pool
    pool = Pool(4)
    file_list = (os.listdir("./real_fake/faces/"))
    r = pool.map(readimg, file_list)
    # print(file_list[:200])
    r = [x for x in r if x != None]
    # test_data, test_labels = zip(*r)
   
    test_data, test_data_gray = zip(*r)
    # print('how do you get labels of', test_labels)
    self.test_data = np.array(test_data)
    # self.test_labels = np.zeros((len(test_labels), 1001))
    self.test_data_gray = np.array(test_data_gray)
    # print('what is happening', self.test_data_gray.shape)
    # self.test_labels[np.arange(len(test_labels)), test_labels] = 1
    # print('2 imagenet image shape ', self.test_data.shape)
    # print('2 imagenet image label shape ', self.test_labels.shape)


class Face_HashModel:
    def __init__(self, hash, bits, factor):
        self.num_channels = 1
        self.image_width = 224
        self.image_height = 224
        # self.num_labels = 10
        global global_threshold
        global_threshold = hash
        global global_bits
        global_bits = bits
        global global_factor
        global_factor = factor
   
    def predict1(self, data, data1, method  ):

        if method == 'linear':
            return tf.py_function(pfunction, [data, data1], tf.float32)
        elif method == 'square':
            return tf.py_function(pfunction_square, [data, data1], tf.float32)
        elif method =='tanh':
            return tf.py_function(pfunction_tanh, [data, data1], tf.float32)
        elif method == 'sigmoid':
            return tf.py_function(pfunction_sigmoid, [data, data1], tf.float32)

    def predict2(self, data, data1):
        return tf.py_function(pfunction_tanh2, [data, data1], tf.float32)

    def predict3(self, data, data1, method):
        return tf.py_function(pfunction_tanh3, [data, data1], tf.float32)

    