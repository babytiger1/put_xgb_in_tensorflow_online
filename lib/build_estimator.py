"""
Build feature columns using tf.feature_column API.
Build estimator using tf.estimator API and custom API (defined in lib module)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

import numpy as np
import tensorflow as tf

# fix ImportError: No mudule named lib.*
import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
from lib.joint import BuildClassifier

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
CATE_STATS_NAME_PREFIX = './data_stat/cate_feat_col_'
DENSE_STATS_NAME = './data_stats/dense_stat.csv'

def build_model_columns():
    """
    Build Categorical and Dense feature columns from custom feature conf using tf.feature_column API
    wide_columns: category features + cross_features + [discretized continuous features]
    deep_columns: continuous features + category features(onehot or embedding for sparse features) + [cross_features(embedding)]
    Return: 
        _CategoricalColumn and __DenseColumn instance in tf.feature_column API
    """
    def parse_dense_stats():
        # get min and max list
        df = pd.read_csv('./data_stat/dense_stat.csv')
        min_row = df.iloc[[3]]  # 'min' row
        max_row = df.iloc[[7]]  # 'max' row
        per75_row = df.iloc[[6]]  # 75% of max
        min_row = np.array(min_row).tolist()[0][2:]  # [0] of [[]]
        max_row = np.array(max_row).tolist()[0][2:]  # pop 'min'string, 'click' label, replace this index to 1
        per75_row = np.array(per75_row).tolist()[0][2:]
        min_max_tup = zip(min_row, max_row)
        return  min_max_tup,per75_row

    def embedding_dim(dim):
        """empirical embedding dim"""
        return int(np.power(2, np.ceil(np.log(dim**0.25))))

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

    dense_columns = [
        'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11',
        'D12', 'D13'
    ]
    cate_columns = [
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
        'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
        'C22', 'C23', 'C24', 'C25', 'C26'
    ]

    cate_columns_dims = []
    for feature in cate_columns:
        with open(CATE_STATS_NAME_PREFIX+feature) as file:
            i = 0
            for _ in file:
                i += 1
            cate_columns_dims.append(i)

    wide_columns = set()
    deep_columns = set()
    emb_columns = set()
    wide_dim = 0
    deep_dim = 0
    for feature, dim in zip(cate_columns, cate_columns_dims):
        col = categorical_column_with_vocabulary_file(
            feature,
            vocabulary_file=CATE_STATS_NAME_PREFIX+feature,
            vocabulary_size=dim,
            default_value=-1,
            num_oov_buckets=0)

        if dim<100:
            wide_columns.add(col)
            wide_dim += dim

        embed_dim = embedding_dim(dim)
        deep_columns.add(
            embedding_column(
                col,
                dimension=embed_dim,
                combiner='mean',
                initializer=None,
                ckpt_to_load_from=None,
                tensor_name_in_ckpt=None,
                max_norm=None,
                trainable=True))
        emb_columns.add(
            embedding_column(
                col,
                dimension=10,
                combiner='mean',
                initializer=None,
                ckpt_to_load_from=None,
                tensor_name_in_ckpt=None,
                max_norm=None,
                trainable=True))
        deep_dim += embed_dim

    min_max_tup, per75_row = parse_dense_stats()
    for feature, min_max, per75_row in zip(dense_columns, min_max_tup,per75_row):
        if per75_row - min_max[0] < (min_max[1] - min_max[0]) * 0.1:   #75 percent of data concentrate in fornt
            normalizer_fn = normalizer_fn_builder('log', min_max)
        else:
            normalizer_fn = normalizer_fn_builder('min_max', min_max)
        col = numeric_column(
            feature,
            shape=(1, ),
            default_value=0,
            dtype=tf.float32,
            normalizer_fn=normalizer_fn)
        wide_columns.add(
            bucketized_column(col, boundaries=[0,0.001, 0.2, 0.4, 0.6, 0.8]))
        deep_columns.add(col)
        wide_dim += 7
        deep_dim += 1

    # add columns logging info
    tf.logging.info('Build total {} embedding columns'.format(
        len(emb_columns)))
    tf.logging.info('Build total {} wide columns'.format(len(wide_columns)))
    tf.logging.info('Build total {} deep columns'.format(len(deep_columns)))
    tf.logging.info('Wide input dimension is: {}'.format(wide_dim))
    tf.logging.info('Deep input dimension is: {}'.format(deep_dim))

    return wide_columns, deep_columns, emb_columns


def build_xdfm_estimator(model_dir):
    """Build an estimator using custom XDFMClassifier API. XDFM = w&d + CIN
    Args:
        model_dir: model save base directory
        model_type: one of {`wide`, `deep`, `wide_deep`, `xdfm`}
    Returns:
        model instance of lib.joint.XDFMClassifier class
    """
    wide_columns, deep_columns, emb_columns = build_model_columns()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        **CONF.runconfig).replace(session_config=config)

    #run_config = tf.estimator.RunConfig(**CONF.runconfig)
    return BuildClassifier(
        model_type='wide_deep',
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        linear_optimizer=CONF.model["linear_optimizer"],
        dnn_feature_columns=deep_columns,
        dnn_optimizer=CONF.model["dnn_optimizer"],
        dnn_hidden_units=CONF.model["dnn_hidden_units"],
        dnn_connected_mode=CONF.model["dnn_connected_mode"],
        emb_feature_columns=emb_columns,
        n_classes=2,
        weight_column=None,
        label_vocabulary=None,
        input_layer_partitioner=None,
        config=run_config)

def build_dcn_estimator(model_dir):
    _, deep_columns, _ = build_model_columns()
    run_config = tf.estimator.RunConfig(**CONF.runconfig)
    return BuildClassifier(
        model_type='deep_cross',
        model_dir=model_dir,
        dnn_feature_columns=deep_columns,
        dnn_optimizer=CONF.model["dnn_optimizer"],
        dnn_hidden_units=CONF.model["dnn_hidden_units"],
        dnn_connected_mode=CONF.model["dnn_connected_mode"],
        cross_feature_columns = deep_columns,
        cross_optimizer = CONF.model["cross_optimizer"],
        n_classes=2,
        config=run_config)

def build_deep_layers(x0, params):
    net = x0
    print(params['hidden_units'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    return net





# def dcn_model_fn(features, labels, mode, params):
#     x0 = tf.feature_column.input_layer(features, params['feature_columns'])
#     print('*' * 100)
#     print('input dim:', x0)
#     last_deep_layer = build_deep_layers(x0, params)
#     print('deep output:', last_deep_layer)
#     last_cross_layer = build_cross_layers(x0, params)
#     print('cross output:', last_cross_layer)
#     last_layer = tf.concat([last_cross_layer, last_deep_layer], 1)
#     my_head = tf.contrib.estimator.binary_classification_head()
#     logits = tf.layers.dense(last_layer, units=my_head.logits_dimension)
#     print('logits', logits)
#     optimizer = tf.train.AdagradOptimizer(
#         learning_rate=params['learning_rate'])
#
#     return my_head.create_estimator_spec(
#         features=features,
#         mode=mode,
#         labels=labels,
#         logits=logits,
#         train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
#     )
