from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
import tensorflow as tf
import tensorpack.tfutils.symbolic_functions as symbf

if sys.version_info.major == 3:
    xrange = range
  
def im2uint8(x):
    x = (x+1.0)*127.5
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8)
    else:
        t = np.clip(x, 0, 255)
        return t.astype(np.uint8)

def scaled_psnr(x, y, name=None):
    x = (x+1.0)*127.5
    y = (y+1.0)*127.5
    return symbf.psnr(x, y, 255, name=name)

def load_checkpoint(sess, checkpoint_dir, saver):
    print(" [*] Loading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        saver.restore(sess, ckpt_path)
        print(" [*] Loading successful!")
        return ckpt_path
    else:
        print(" [*] No suitable checkpoint!")
        return None
