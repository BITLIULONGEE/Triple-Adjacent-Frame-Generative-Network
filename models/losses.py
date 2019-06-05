# Copyright (c) 2016-2017 Shafeen Tejani. Released under GPLv3.
import tensorflow as tf
import numpy as np
from utils import vgg_net

FEATURE_LAYER = 'conv3_3'
VGG_PATH = './vgg19/imagenet-vgg-verydeep-19.mat'
vgg = vgg_net.VGG(VGG_PATH)

def l2_loss(a, b, scope=None):
    with tf.name_scope(scope, 'l2_loss', [a, b]):
        loss = tf.reduce_mean((a - b) ** 2) 
        return loss

def l1_loss(a, b, scope=None):
    with tf.name_scope(scope, 'l1_loss', [a, b]):
        loss = tf.reduce_mean(tf.abs(a - b))
        return loss

def perception_loss(estimated_batch, target_batch,  weights=1.0, name=None):
    target_net = vgg.net(target_batch)
    estimated_net = vgg.net(estimated_batch)
    return l1_loss(estimated_net[FEATURE_LAYER], target_net[FEATURE_LAYER])* weights
    
def Pixel_loss(estimated_batch, target_batch,  weights=1.0,  name=None):
    estimated_batch, target_batch = (estimated_batch + 1.0)*127.5, (target_batch + 1.0)*127.5
    return l2_loss(estimated_batch, target_batch) * weights

def tv_loss(self, image,  weights=1.0, name=None):
    # total variation denoising
    shape = tuple(image.get_shape().as_list())
    tv_y_size = _tensor_size(image[:,1:,:,:])
    tv_x_size = _tensor_size(image[:,:,1:,:])
    tv_loss = l2_loss(image[:,1:,:,:], image[:,:shape[1]-1,:,:])/tv_y_size + l2_loss(image[:,:,1:,:], image[:,:,:shape[2]-1,:])/tv_x_size
    return tv_loss * weights
