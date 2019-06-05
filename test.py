#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models.model as model
from utils.data import *
from utils.util import *

import os
import argparse
import numpy as np
import time
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Tripledeblur arguments')
    parser.add_argument('--name', type=str, default='TripleFrameDeblur', help='name of the expriment')
    parser.add_argument('--phase', type=str, default='test', help='determine whether train or test')
    parser.add_argument('--batchSize', help='training batch size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--cropSize', help='crop to patch', type=int, default=256)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='1', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=600, help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=800, help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--input_path', type=str, required=True, help='input path for testing images')
    parser.add_argument('--output_path', type=str, required=True, help='output path for testing images')
    args = parser.parse_args()
    return args

def main(_):
    args = parse_args()
    # set gpu/cpu mode
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
    # set up deblur models
    M = model.DEBLUR(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with tf.Graph().as_default():
        before = tf.placeholder(tf.float32, shape=[1, args.height, args.width, 3])
        current = tf.placeholder(tf.float32, shape=[1, args.height, args.width, 3])
        after = tf.placeholder(tf.float32, shape=[1, args.height, args.width, 3])
            
        predict = M.generator(before,current,after,istrain=False)

        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            checkpoint_dir = os.path.join('./checkpoints/', args.name)
            load_checkpoint(sess, checkpoint_dir, saver)
            imgsName = sorted(os.listdir(args.input_path))
            time_start = time.time()
            for imgName in imgsName:
                print('Processing: %s' % imgName)
	        img = cv2.imread(os.path.join(args.input_path, imgName))
                inputs = split_input(img)
                inputs = [np.expand_dims(img, 0) for img in inputs]
               
                result = sess.run(predict, feed_dict={before:inputs[0], current:inputs[1], after:inputs[2]})
               
                result = result[1][0,:,:,:]
                result = im2uint8(result)
                cv2.imwrite(os.path.join(args.output_path, imgName), result)
            time_end = time.time()
            process_time = (time_end-time_start)
            print('avage process time = %.4f'%(process_time/len(imgsName)))
                    
if __name__ == '__main__':
    tf.app.run()
