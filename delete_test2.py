from __future__ import division, print_function, absolute_import
import tensorflow as tf
import xgb_model_zzr2
import testvars2
import tensorflow as tf
layers = tf.keras.layers
from tensorflow.contrib import autograph




print(autograph.to_code(testvars2.XGBprocess))
