from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

import numpy as np
import tensorflow as tf
from lib.read_conf import Config
# fix ImportError: No mudule named lib.*
import sys
import xgb_model_zzr
import xgb2tensorflow
import datasets
import testvars4
import yaml


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

# wide columns
categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
categorical_column_with_vocabulary_file = tf.feature_column.categorical_column_with_vocabulary_file
crossed_column = tf.feature_column.crossed_column
bucketized_column = tf.feature_column.bucketized_column
# deep columns
embedding_column = tf.feature_column.embedding_column
indicator_column = tf.feature_column.indicator_column
numeric_column = tf.feature_column.numeric_column

CONF = Config()

def get_params(colName):
    names = CONF.read_feature_conf()
    return names[colName]


def normalizer_fn_builder(scaler, normalization_params):
    """normalizer_fn builder"""
    if scaler == 'min_max':
        if normalization_params[0] == normalization_params[1]:
            return lambda x: x - x
        else:
            # return lambda x: 0 if x == 0 else 0
            return lambda x: (x - normalization_params[0]) / (normalization_params[1] - normalization_params[0])
    elif scaler == 'standard':
        return lambda x: (x - normalization_params[0]) / normalization_params[1]
    else:
        return lambda x: tf.log1p(x - normalization_params[0]) / tf.log1p(normalization_params[1] - normalization_params[0])


def transform_feature():
    conFeatureName = CONF.get_feature_name(feature_type = 'used')
    conFeatureCol = []
    catFeatureName=[]
    FeatureCol_forxgb=[]
    catFeatureCol = []
    conFeatureName.remove('label')
    for elemName in conFeatureName:
        params = get_params(elemName)
        conFeatureCol.append(numeric_column(elemName,dtype=tf.float32,normalizer_fn=normalizer_fn_builder('min_max',params)) )
        FeatureCol_forxgb.append(numeric_column(elemName, dtype=tf.float32))

    for elemName in catFeatureName:
        catFeatureCol.append(categorical_column_with_identity(elemName))
        FeatureCol_forxgb.append(categorical_column_with_identity(elemName))

    return conFeatureCol, FeatureCol_forxgb,catFeatureCol


# try to transform x into <str:x>
def transform_type(x,elemName):
    return tf.map_fn(lambda x:x,x)

if __name__ == '__main__':
    f1,f2,f3 = transform_feature()
    print(f1)
    print(f2)
    print(f3)