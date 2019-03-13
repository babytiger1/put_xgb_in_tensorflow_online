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

def transform_XGB_Model(x,allFeature):
	
	res = xgb_model_zzr.xgb_predict(dictDate)
	return res

def transform2cat():
	#allFeatureName = ["var01","var02","var03","var04","var05","var06","var07","var08","var09","var10","var11","var12","var13","var14","var15","var16","var17","var18","var19","var20","var21","var22","var23","var24","var25","var26","var27","var28","var29","var30","var31","var32","var33","var34","var35","var36","var37","var38","var39","var40","var41","var42","var43","var44","var45","var46","var47","var48","var49","var50","var51","var52","var53","var54","var55","var56","var57","var58","var59","var60"]
	#for col in catColumn:
	#	allFeature.append( categorical_column_with_identity(col) )
	#for col in conColumn:
	#	allFeature.append( numeric_column(col) )
	resFeature = numeric_column('allFeatures',default_value=0,dtype=tf.float32,normalizer_fn=lambda x:xgb_model_zzr.xgb_predict(x))

def concat_all_feature(x):
    allFeatureName = ["var01","var02","var03","var04","var05","var06","var07","var08","var09","var10","var11","var12","var13","var14","var15","var16","var17","var18","var19","var20","var21","var22","var23","var24","var25","var26","var27","var28","var29","var30","var31","var32","var33","var34","var35","var36","var37","var38","var39","var40","var41","var42","var43","var44","var45","var46","var47","var48","var49","var50","var51","var52","var53","var54","var55","var56","var57","var58","var59","var60"]
    data = dict(zip(allFeatureName,x))
    return json.dumps(data)

if __name__ == '__main__':
    data = pandas.read_csv("test_data1.csv")
    data['allFeatures'] = data.apply(lambda x:concat_all_feature(x),axis = 1)
    tensor = tf.feature_column.input_layer(data['allFeatures'].values,transform2cat() )



data = pandas.read_csv("test_data1.csv")
data['allFeatures'] = data.apply(lambda x:concat_all_feature(x),axis = 1)
resFeature = numeric_column('allFeatures',default_value=0,dtype=tf.string,normalizer_fn=lambda x:xgb_model_zzr.xgb_predict(x))
tensor = tf.feature_column.input_layer({'allFeatures':data['allFeatures'].values}, resFeature )

###############fail##################################################################


import tensorflow as tf

constan = tf.constant( value = [5.0,8.0], dtype=tf.float32 , name = 'CONSTANT')
varia = tf.placeholder( name = 'varia' , shape=[1,2], dtype=tf.float32)

resaa = tf.cond(tf.math.logical_and(varia[0]>constan[0], varia[1]>constan[1]) , lambda:tf.add(varia,constan),    lambda:tf.square(varia))
#resaa = tf.case((varia[0] > constan[0], varia[1]>constan[1]) , lambda:tf.add(varia,constan),lambda:tf.square(varia))


with  tf.Session() as  sess:
    s = sess.run(resaa,feed_dict={varia:[[10,20]]})
    print(s)
    inputs = {"varia":varia}
    outputs = {"resaa":resaa}
    tf.saved_model.simple_save(sess,'./saved_model_forxgb2',inputs,outputs)
    LOGDIR = './logdir'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    train_writer.flush()
    train_writer.close()



##################################fail#################################################


import tensorflow as tf
import xgb_model_zzr

from __future__ import division, print_function, absolute_import
  
import tensorflow as tf
layers = tf.keras.layers
from tensorflow.contrib import autograph




print(autograph.to_code(xgb_tree))














