#!/usr/bin/env python
# -*- coding: utf-8 -*-
# name: Triple Images Motion Deblurring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from models.GAN import GANModelDesc
from models.losses import *
from utils.util import *

class DEBLUR(GANModelDesc):
    def __init__(self, args):
        self.args = args
        self.cropSize = args.cropSize
        self.batchSize = args.batchSize
        self.learning_rate = args.learning_rate
        self.dis_nf = 64
        self.lambda1 = 20
        self.lambda2 = 100

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, self.cropSize, self.cropSize, 3), 'before'),
                tf.placeholder(tf.float32, (None, self.cropSize, self.cropSize, 3), 'current'),
                tf.placeholder(tf.float32, (None, self.cropSize, self.cropSize, 3), 'after'),
                tf.placeholder(tf.float32, (None, self.cropSize, self.cropSize, 3), 'sharp')]
    
    def generator(self, img1, img2, img3, istrain):
        def INReLU(x, name=None):
            x = InstanceNorm('lns', x)
            return tf.nn.relu(x)
 
        def ResnetBlock(name, x, channels, split=1):
            with tf.variable_scope(name):
                y = Conv2D('cv1', x, channels, kernel_size=3, strides=1, split=split)
                y = Dropout('dp', y)     
                y = Conv2D('cv2', y, channels, kernel_size=3, strides=1, activation=None, split=split)
                y = InstanceNorm('in', y)
                x+y
            return x+y
                
        def channel_shuffle(x, num_grounp):
            n, h, w, c = x.get_shape().as_list()
            x_reshaped = tf.reshape(x,[-1,h,w,num_grounp,c//num_grounp])
            x_transposed = tf.transpose(x_reshaped,[0,1,2,4,3])
            output = tf.reshape(x_transposed,[-1,h,w,c])
            return output
        
        def Temporal_Merge(name, incoming_skip, tensor):
            with tf.variable_scope(name):
                channels = tensor.get_shape().as_list()[3]
                tensor1, tensor2 = tf.split(tensor, num_or_size_splits=2, axis=3)
                top = tf.concat([tensor1, incoming_skip], 3)
                bottom = tf.concat([tensor2, incoming_skip], 3)
                top = Conv2D('top1' , top, channels/2, kernel_size=1)
                top = Conv2D('top2', top, channels, kernel_size=3)
                top = Conv2D('top4', top, channels/2, kernel_size=1)
                bottom = Conv2D('bottom1', bottom, channels/2, kernel_size=1)
                bottom = Conv2D('bottom2', bottom, channels, kernel_size=3)
                bottom = Conv2D('bottom3', bottom, channels/2, kernel_size=1)
                tensor_internal = tf.concat([top, bottom], 3)
                tensor_internal = channel_shuffle(tensor_internal,8)
                return tensor_internal
        
        with tf.variable_scope('gen'):
            with argscope([Conv2D, Conv2DTranspose], activation=INReLU),\
                 argscope(Dropout, is_training=istrain):  
                fd_conv1_1 = Conv2D('fd_conv1_1', img2, 64, kernel_size=7)
                fd_res1_1 = ResnetBlock('fd_res1_1', fd_conv1_1, 64)
                fd_res1_2 = ResnetBlock('fd_res1_2', fd_res1_1, 64)
                
                fd_conv2_1 = Conv2D('fd_conv2_1',fd_res1_2, 128, kernel_size=5, strides=2)
                fd_res2_1 = ResnetBlock('fd_res2_1', fd_conv2_1, 128)
                fd_res2_2 = ResnetBlock('fd_res2_2', fd_res2_1, 128)
                   
            	fd_conv3_1 = Conv2D('fd_conv3_1', fd_res2_2, 256, kernel_size=5, strides=2)
            	fd_res3_1 = ResnetBlock('fd_res3_1', fd_conv3_1, 256)
                fd_res3_2 = ResnetBlock('fd_res3_2', fd_res3_1, 256)
                fd_res3_3 = ResnetBlock('fd_res3_3', fd_res3_2, 256)
            	    
            	fd_res4_1 = ResnetBlock('fd_res4_1', fd_res3_3, 256)
                fd_res4_2 = ResnetBlock('fd_res4_2', fd_res4_1, 256)
                fd_res4_3 = ResnetBlock('fd_res4_3', fd_res4_2, 256)
            	fd_deconv1_1 = Conv2DTranspose('fd_deconv1_1', fd_res4_3, 128, kernel_size=4, strides=2)   
                fd_deconv1_1 =  fd_deconv1_1 + fd_res2_2

                fd_res5_1 = ResnetBlock('fd_res5_1', fd_deconv1_1, 128)
                fd_res5_2 = ResnetBlock('fd_res5_2', fd_res5_1, 128)
                fd_deconv2_1 = Conv2DTranspose('fd_deconv2_1', fd_res5_2, 64, kernel_size=4, strides=2)  
                fd_deconv2_1 =  fd_deconv2_1 + fd_res1_2

                fd_res6_1 = ResnetBlock('fd_res6_1', fd_deconv2_1, 64)
                fd_res6_2 = ResnetBlock('fd_res6_2', fd_res6_1, 64)
                fd_deconv3_1 = Conv2D('fd_deconv3_1', fd_res6_2, 12, kernel_size=5)
                fd_deconv3_2 = Conv2D('fd_deconv3_2', fd_deconv3_1, 3, kernel_size=3, activation=tf.nn.tanh)
                inp1 = fd_deconv3_2 + img2
                
                #-------------------------------------------------------------------------------------------
                #-------------------------------------------------------------------------------------------
                
                inputs = tf.concat([img1, inp1, inp1, img3], axis=3)
                sd_conv1_1 = Conv2D('sd_conv1_1', inputs, 128, kernel_size=7, split=2)
                sd_merge1 = Temporal_Merge('sd_merge1', fd_deconv2_1, sd_conv1_1) 
                sd_res1_1 = ResnetBlock('sd_res1_1', sd_merge1, 128, split=2)
                sd_res1_2 = ResnetBlock('sd_res1_2', sd_res1_1, 128, split=2)

                sd_skip1 = Conv2D('sd_skip1', sd_res1_2, 64, kernel_size=1) 
   
                sd_conv2_1 = Conv2D('sd_conv2_1',sd_res1_2, 256, kernel_size=5, strides=2, split=2)
                sd_merge2 = Temporal_Merge('sd_merge2', fd_deconv1_1, sd_conv2_1) 
                sd_res2_1 = ResnetBlock('sd_res2_1', sd_merge2, 256, split=2)
                sd_res2_2 = ResnetBlock('sd_res2_2', sd_res2_1, 256, split=2)
                
                sd_skip2 = Conv2D('sd_skip2', sd_res2_2, 128, kernel_size=1) 

            	sd_conv3_1 = Conv2D('sd_conv3_1', sd_res2_2, 512, kernel_size=5, strides=2, split=2)
                sd_merge3 = Temporal_Merge('sd_merge3',fd_res3_3, sd_conv3_1)
            	sd_res3_1 = ResnetBlock('sd_res3_1', sd_merge3, 512, split=2)
                sd_res3_2 = ResnetBlock('sd_res3_2', sd_res3_1, 512, split=2)
                sd_res3_3 = ResnetBlock('sd_res3_3', sd_res3_2, 512, split=2)
                    
                sd_deconv1_1 = Conv2D('sd_deconv1_1', sd_res3_3, 256, kernel_size=1)
            	sd_res4_1 = ResnetBlock('sd_res4_1', sd_deconv1_1, 256)
                sd_res4_2 = ResnetBlock('sd_res4_2', sd_res4_1, 256)
                sd_res4_3 = ResnetBlock('sd_res4_3', sd_res4_2, 256)
            	sd_deconv1_2 = Conv2DTranspose('sd_deconv1_2', sd_res4_3, 128, kernel_size=4, strides=2)
                sd_deconv1_2 =  sd_deconv1_2 + sd_skip2

                sd_res5_1 = ResnetBlock('sd_res5_1', sd_deconv1_2, 128)
                sd_res5_2 = ResnetBlock('sd_res5_2', sd_res5_1, 128)
                sd_deconv2_1 = Conv2DTranspose('sd_deconv2_1', sd_res5_2, 64, kernel_size=4, strides=2)  
                sd_deconv2_1 =  sd_deconv2_1 + sd_skip1

                sd_res6_1 = ResnetBlock('sd_res6_1',sd_deconv2_1, 64)
                sd_res6_2 = ResnetBlock('sd_res6_2', sd_res6_1, 64)
                sd_deconv3_1 = Conv2D('sd_deconv3_1', sd_res6_2, 12, kernel_size=5)
                sd_deconv3_2 = Conv2D('sd_deconv3_2', sd_deconv3_1, 3, kernel_size=3, activation=tf.nn.tanh)
                inp2 = sd_deconv3_2 + inp1
                return inp1, inp2

    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        with tf.variable_scope('discrim'):
            with argscope(Conv2D, activation=tf.identity, kernel_size=4, strides=2): 
                l = (LinearWrap(imgs)
                     	.Conv2D('conv0', self.dis_nf, activation=tf.nn.leaky_relu)
                 	.Conv2D('conv1', self.dis_nf * 2)
                 	.InstanceNorm('ln1')
                 	.tf.nn.leaky_relu()
                 	.Conv2D('conv2', self.dis_nf * 4)
                 	.InstanceNorm('ln2')
                 	.tf.nn.leaky_relu()
                 	.Conv2D('conv3', self.dis_nf * 8)
                 	.InstanceNorm('ln3')
                 	.tf.nn.leaky_relu()
                        .Conv2D('conv4', self.dis_nf * 8)
                 	.InstanceNorm('ln4')
                 	.tf.nn.leaky_relu()
                 	.FullyConnected('fct', 1, activation=None)())
                return tf.reshape(l, [-1])

    # deblur-motion
    def build_graph(self, before, current, after, target):
        
        with argscope([Conv2D, Conv2DTranspose, FullyConnected],
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            
            internal, estimate = self.generator(before, current, after, istrain=True)
          
            # the Wasserstein-GAN losses
            alpha = tf.random_uniform(shape=[2*self.batchSize, 1, 1, 1], minval=0., maxval=1.)
            Twotarget = tf.concat([target, target],0)
            Twoestimate = tf.concat([internal, estimate], 0)
            Twointerp = Twotarget + alpha * (Twoestimate - Twotarget)

            real_pred = self.discriminator(Twotarget)
            fake_pred = self.discriminator(Twoestimate)
            vec_interp = self.discriminator(Twointerp)

        # the Wasserstein-GAN losses
        d_loss = tf.reduce_mean(fake_pred - real_pred)
        g_loss = tf.negative(tf.reduce_mean(fake_pred))

        # the gradient penalty loss
        gradients = tf.gradients(vec_interp, Twointerp)[0]
        gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(gradients - 1), name='gradient_penalty')
        
        psnr_base = tf.identity(scaled_psnr(current, target),name='PSNR_BASE')

        psnr_1 = scaled_psnr(internal, target, name='PSNR_1')
        psnr_impro1 = tf.divide(psnr_1, psnr_base, name='PSNR_IMPRO1')

        psnr_2 = scaled_psnr(estimate, target, name='PSNR_2')
        psnr_impro2 = tf.divide(psnr_2, psnr_base, name='PSNR_IMPRO2')
        add_moving_summary(psnr_base, psnr_1,psnr_2, psnr_impro1,  psnr_impro2)
       
        #pixel_loss
        pixel_loss_1 = tf.identity(Pixel_loss(internal, target, self.lambda1), 'pixel_loss1')
        pixel_loss_2 = tf.identity(Pixel_loss(estimate, target, self.lambda1), 'pixel_loss2')
        pixel_loss = tf.add(pixel_loss_1, pixel_loss_2, name='pixel_loss')
        add_moving_summary(pixel_loss_1, pixel_loss_2, pixel_loss)
        
        # the perception losses
        internal_feature_loss = tf.identity(perception_loss(internal, target, self.lambda2), 'feature_loss1')
        estimate_feature_loss = tf.identity(perception_loss(estimate, target, self.lambda2), 'feature_loss2')
        feature_loss = tf.add(internal_feature_loss, estimate_feature_loss, name='feature_loss')
        add_moving_summary(internal_feature_loss, estimate_feature_loss, feature_loss)

        content_loss = tf.add(feature_loss, pixel_loss, name='content_loss')
        add_moving_summary(content_loss)

        self.d_loss = tf.add(d_loss, 10*gradient_penalty, name='d_loss')
        self.g_loss = tf.add(g_loss, content_loss, name='g_loss')
        add_moving_summary(self.g_loss, self.d_loss, gradient_penalty)
        
        # just visualize original images
        viz1 = tf.concat([before, current, after], axis=2)
        viz2 = tf.concat([internal, estimate, target], axis=2)
        viz = tf.concat([viz1, viz2], axis=1)
        tf.summary.image('viz', im2uint8(viz), max_outputs=30)

        self.collect_variables()

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.learning_rate, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5)

    
