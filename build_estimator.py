from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

import numpy as np
import tensorflow as tf

# fix ImportError: No mudule named lib.*
import sys
import xgb_model_zzr
import xgb2tensorflow
import datasets
import make_column_feature
import testvars4
from dnn_classifier import BuildDNNClassifier
from lib.read_conf import Config

CONF = Config()

def build_dnn_estimator(model_dir,mode_type):
    features_columns = make_column_feature.transform_feature()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #run_config = tf.estimator.RunConfig(
    #    **CONF.runconfig).replace(session_config=config)
    if mode_type == 'dnn':
        return BuildDNNClassifier(
            model_dir=CONF.train['model_dir'] ,
            features_columns = features_columns,
            dnn_connected_mode= CONF.model['dnn_connected_mode'],
            dnn_optimizer=CONF.model['dnn_optimizer'],#CONF.model["dnn_optimizer"],
            dnn_hidden_units=CONF.model["dnn_hidden_units"],
            n_classes=2,
            weight_column=None,
            label_vocabulary=None,
            input_layer_partitioner=None)


if __name__ == '__main__':
    build_dnn_estimator("/logs", "dnn")


