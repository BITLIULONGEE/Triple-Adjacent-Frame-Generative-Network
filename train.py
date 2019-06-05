#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import argparse
from tensorpack import *
from tensorpack.utils import logger
from models.GAN import SeparateGANTrainer

import models.model as model
from utils.data import *
from utils.util import *

def parse_args():
    parser = argparse.ArgumentParser(description='Tripledeblur arguments')
    parser.add_argument('--name', type=str, default='TripleFrameDeblur', help='name of the expriment')
    parser.add_argument('--batchSize', help='training batch size', type=int, default=1)
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--load', help='path to load model')
    parser.add_argument('--cropSize', help='crop to patch', type=int, default=256)
    parser.add_argument('--max_epoch', help='max number of epoch', type=int, default=600)
    parser.add_argument('--steps_per_epoch', help='steps_per_epoch', type=int, default=5000)
    parser.add_argument('--continue_train', action='store_true', help='if specified, continue to train')
    args = parser.parse_args()
    return args

def main(_):
    args = parse_args()
    # set gpu/cpu mode
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    checkpoint_dir = os.path.join('./checkpoints/',args.name)
    logger.set_logger_dir(checkpoint_dir)
    
    # set up deblur models
    M = model.DEBLUR(args)
    
    ds_train = get_data(args.dataroot, phase='train', crop_size=args.cropSize, batch_size=args.batchSize)
    ds_val = get_data(args.dataroot, phase='val', crop_size=args.cropSize, batch_size=args.batchSize)

    trainer = SeparateGANTrainer(ds_train, M, g_period=6)
    trainer.train_with_defaults(
                 callbacks=[ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),
                            ScheduledHyperParamSetter('learning_rate', [(300, args.learning_rate), (args.max_epoch, 0)], interp='linear'),
                            InferenceRunner(ds_val, [ScalarStats('PSNR_BASE'),ScalarStats('PSNR_2'), ScalarStats('PSNR_IMPRO2'),ScalarStats('pixel_loss2'), ScalarStats('feature_loss2')])],
                 session_init=SaverRestore(checkpoint_dir+'/model-431249.data-00000-of-00001') if args.continue_train else None,
                 starting_epoch=1,
                 steps_per_epoch=args.steps_per_epoch, 
                 max_epoch=args.max_epoch)

if __name__ == '__main__':
    tf.app.run()
