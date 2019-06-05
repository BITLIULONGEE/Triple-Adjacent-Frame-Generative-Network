from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorpack import *
from glob import glob
import os

def split_input(img):      # split the image into 3*blurrys + target 
    img = img/ 127.5 - 1.0
    #h = img.shape[0] // 2
    #w = img.shape[1] // 2
    #blurry = [img[:h, :w, :], img[:h, w:, :], img[h:, :w, :]]
    #sharp = [img[h:, w:, :]]
    #return blurry + sharp
    h = img.shape[0] // 3
    blurry = [img[:h, :, :], img[h:2*h, :, :], img[2*h:, :, :]]
    return blurry         

def get_data(datadir, phase='train', crop_size=256, batch_size=1):

    imgs = glob(os.path.join(datadir, phase, '*.jpg'))
    if phase == 'train':
        ds = ImageFromFile(imgs, channel=3, shuffle=True)
        ds = MapData(ds, lambda dp: split_input(dp[0]))
        ds = AugmentImageComponents(ds, [imgaug.RandomCrop(crop_size)], index=range(4), copy=True)
        ds = RandomMixData([ds])
        ds = BatchData(ds, batch_size)
        ds = PrefetchDataZMQ(ds, 12)
        ds = QueueInput(ds)
    else: 
        ds = ImageFromFile(imgs, channel=3, shuffle=False)
        ds = MapData(ds, lambda dp: split_input(dp[0]))
        ds = AugmentImageComponents(ds, [imgaug.CenterCrop(crop_size)], index=range(4), copy=True)
        ds = RandomMixData([ds])
        ds = BatchData(ds, 1)
        ds = FixedSizeData(ds, 600)
        ds = PrefetchDataZMQ(ds, 12)
        ds = QueueInput(ds)
    return ds
