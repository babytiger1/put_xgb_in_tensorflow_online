from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

from collections import OrderedDict
import tensorflow as tf
from lib.read_conf import Config
from copy import deepcopy

import  testvars4

# fix ImportError: No mudule named lib.*
import sys
import xgb_model_zzr
import xgb2tensorflow

conf = Config()
train_conf = conf.train
num_parallel_calls = train_conf["num_parallel_calls"]
shuffle_buffer_size = train_conf["num_examples"]
train_epochs = train_conf["train_epochs"]

use_weight = False
feature =conf.get_feature_name()  # all features
feature_used = conf.get_feature_name('used')  # used features
feature_unused = conf.get_feature_name('unused')  # unused features
feature_conf = conf.read_feature_conf()  # feature conf dict
csv_defaults_values =  [0.0]*31+[0.0]
feature_name = ["id","vars0","vars1","vars2","vars3","vars4","vars5","vars6","vars7","vars8","vars9","vars10","vars11","vars12","vars13","vars14","vars15","vars16","vars17","vars18","vars19","vars20","vars21","vars22","vars23","vars24","vars25","vars26","vars27","vars28","vars29","label"]
# self._multivalue = self._train_conf["multivalue"]

#
# csv_defaults_keys = ["var01", "var02", "var03", "var04", "var05", "var06", "var07", "var08", "var09", "var10", "var11",
#                      "var12", "var13", "var14", "var15", "var16", "var17", "var18", "var19", "var20", "var21", "var22",
#                      "var23", "var24", "var25", "var26", "var27", "var28", "var29", "var30", "var31", "var32", "var33",
#                      "var34", "var35", "var36", "var37", "var38", "var39", "var40", "var41", "var42", "var43", "var44",
#                      "var45", "var46", "var47", "var48", "var49", "var50", "var51", "var52", "var53", "var54", "var55",
#                      "var56", "var57", "var58", "var59", "var60","label"]


def _column_to_csv_defaults():
    """parse columns to record_defaults param in tf.decode_csv func
    Return:
        OrderedDict {'feature name': [''],...}
    """
    csv_defaults = OrderedDict()
    csv_defaults['label'] = [0]  # first label default, empty if the field is must
    for f in feature:
        if f in feature_used:  # used features
            conf = feature_conf[f]
            if conf == '':
                csv_defaults[f] = ['']
            else:
                csv_defaults[f] = [0.0]  # 0.0 for float32
        else:  # unused features
            csv_defaults[f] = ['']
    return csv_defaults


def input_fn(data_file, mode, batch_size):
    assert mode in {
        'train', 'eval', 'pred'
    }, ('mode must in `train`, `eval`, or `pred`, found {}'.format(mode))
    tf.logging.info('Parsing input csv files: {}'.format(data_file))
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)
    # Use `Dataset.map()` to build a pair of a feature dictionary
    # and a label tensor for each example.
    # Shuffle, repeat, and batch the examples.

    #transform_XGB_Model = autograph.to_graph(testvars4.XGBprocess)
    #print(autograph.to_code(testvars4.XGBprocess))


    dataset = dataset.map(_parse_csv(is_pred=(mode == 'pred')),num_parallel_calls=num_parallel_calls)
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=123)
        dataset = dataset.repeat(train_epochs)  # define outside loop
        #dataset = dataset.repeat()
    dataset = dataset.prefetch(2 * batch_size)

    padding_dic = {k: [None] for k in feature_used if k != "label"}
    #padding_dic = {'multivale': [None]}
    padded_shapes = (padding_dic,[None])
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    #dataset = dataset.batch(batch_size)

    # batch(): each element tensor must have exactly same shape, change rank 0 to rank 1
    # dataset = dataset.batch(batch_size)
    a= dataset.make_one_shot_iterator().get_next()
    print(a)
    return dataset.make_one_shot_iterator().get_next()



def _parse_csv(
               is_pred=False,
               field_delim=',',
               na_value='',
               multivalue_delim=',',):
    """Parse function for csv data
    Args:
        is_pred: bool, defaults to False
            True for pred mode, parse input data with label
            False for train or eval mode, parse input data without label
        field_delim: csv fields delimiter, defaults to `\t`
        na_value: use csv defaults to fill na_value
        multivalue: bool, defaults to False
            True for csv data with multivalue features.
            eg:   f1       f2   ...
                a, b, c    1    ...
                 a, c      2    ...
                 b, c      0    ...
        multivalue_delim: multivalue feature delimiter, defaults to `,`
    Returns:
        feature dict: {feature: Tensor ... }
    """
    _csv_defaults = _column_to_csv_defaults()
    if is_pred:
        _csv_defaults.pop('label')
    use_weight = None

    def parser(value):
        """Parse train and eval data with label
        Args:
            value: Tensor("arg0:0", shape=(), dtype=string)
        """
        # `tf.decode_csv` return rank 0 Tensor list: <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>
        # na_value fill with record_defaults
        columns = tf.decode_csv(value,
            record_defaults=list(csv_defaults_values),  # 通过空缺值来判断类型
            field_delim=field_delim,
            use_quote_delim=False,
            na_value=na_value)
        features = dict(zip(feature_name, columns))

        for unused in feature_unused:
            features.pop(unused)
        for used in feature_used:
            features[used] = tf.expand_dims(features[used], 0)

            # for f, tensor in list(features):
        #     if f in feature_unused:
        #         features.pop(f)  # remove unused features
        #         continue
        #     if f == 'multivalue':  # split tensor
        #         # if isinstance(csv_defaults[f][0], str):
        #         # input must be rank 1, return SparseTensor
        #         # print(st.values)  # <tf.Tensor 'StringSplit_11:1' shape=(?,) dtype=string>
        #         # tensor = tf.expand_dims(tensor, 0)
        #         features[f] = tf.string_split(
        #             [tensor], multivalue_delim).values  # tensor shape (?,)
        #         print(features[f].shape)
        #     else:
        #         features[f] = tf.expand_dims(tensor, 0)  # change shape from () to (1,)
        #         # features[f] = tensor

        if is_pred:
            return features
        else:
            labels = features.pop('label')
            if use_weight:
                pred = labels[0]  # pred must be rank 0 scalar
                pos_weight, neg_weight = 1, 1
                weight = tf.cond(pred, lambda: pos_weight,lambda: neg_weight)
                features["weight_column"] = [weight]  # padded_batch need rank 1
            return features, labels

            # features[f] = tensor
        # if is_pred:
        #     return features
        #else:
            # labels = tf.equal(features.pop('label'), 1)
            # if use_weight:
            #     pred = labels[0]  # pred must be rank 0 scalar
            #     pos_weight, neg_weight = 1, 1
            #     weight = tf.cond(pred, lambda: pos_weight,
            #                      lambda: neg_weight)
            #     features["weight_column"] = [weight
            #                                  ]  # padded_batch need rank 1
            # return features, labels
    return parser