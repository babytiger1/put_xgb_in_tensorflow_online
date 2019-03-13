from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf


import xgb_model_zzr
import xgb2tensorflow
import datasets
import make_column_feature
import testvars4
from tensorflow import contrib
autograph = contrib.autograph
from build_estimator import build_dnn_estimator










# def transform_XGB_Model(x):
#     allFeatureName = ["var01","var02","var03","var04","var05","var06","var07","var08","var09","var10","var11","var12","var13","var14","var15","var16","var17","var18","var19","var20","var21","var22","var23","var24","var25","var26","var27","var28","var29","var30","var31","var32","var33","var34","var35","var36","var37","var38","var39","var40","var41","var42","var43","var44","var45","var46","var47","var48","var49","var50","var51","var52","var53","var54","var55","var56","var57","var58","var59","var60"]
#     dictDate = dict(zip(x,allFeatureName))
# 	#res = xgb_model_zzr.xgb_predict(dictDate)
#     res = xgb2tensorflow.tf__XGBprocess(dictDate)
#     return res


# def transform2cat(allFeature):
#     allFeatureName = ["var01","var02","var03","var04","var05","var06","var07","var08","var09","var10","var11","var12","var13","var14","var15","var16","var17","var18","var19","var20","var21","var22","var23","var24","var25","var26","var27","var28","var29","var30","var31","var32","var33","var34","var35","var36","var37","var38","var39","var40","var41","var42","var43","var44","var45","var46","var47","var48","var49","var50","var51","var52","var53","var54","var55","var56","var57","var58","var59","var60"]
#     tfallFeatureName = tf.convert_to_tensor(allFeatureName,dtype=tf.string,name="allFeatureName")
#     fn = lambda x: transform_XGB_Model(x, tfallFeatureName)
#     tensorAfterXgb = tf.map_fn(fn,allFeature)
#     return tf.concat([allFeature,tensorAfterXgb],axis=-1)

    #catColumn = []
    #conColumn = []
	#for col in catColumn:
	#	allFeature.append( categorical_column_with_identity(col) )
	#for col in conColumn:
	#	allFeature.append( numeric_column(col) )


	# resFeature = numeric_column(allFeature,
    #         default_value=0,
    #         dtype=tf.float32,
    #         normalizer_fn=lambda x:normalizer_fn_builder('min_max',x)))








if __name__ == '__main__':
    data = datasets.input_fn("test_data1.csv",mode="train",batch_size=10)
    #print(data)


    #print(conFeatureCol)
    ouputType = [tf.int32]*4

    #model = build_dnn_estimator("/logs", 'dnn')

    #featureTensors = transform2cat(tensor)


    #print(autograph.to_code(testvars4.XGBprocess))

    conFeatureCol, FeatureCol_forxgb, catFeatureCol = make_column_feature.transform_feature()
    transform_XGB_Model = autograph.to_graph(testvars4.XGBprocess4)
    with tf.Session() as sess:
        coninputTensor = tf.feature_column.input_layer(data, conFeatureCol)
        xgbinputTensor = tf.feature_column.input_layer(data, FeatureCol_forxgb)
        catinputTensor = tf.feature_column.input_layer(data, conFeatureCol)
        # print(xgbinputTensor.shape)
        # xgbinputTensor =  tf.transpose(xgbinputTensor,perm=[1,0])
        # print(xgbinputTensor.shape)
        xgbFeature = tf.map_fn(transform_XGB_Model,xgbinputTensor,dtype=ouputType)
        #xgbFeature = transform_XGB_Model(xgbinputTensor)
        #dnnFeature = tf.concat([coninputTensor,catinputTensor,xgbFeature],axis=-1)
        aaa = tf.concat([xgbFeature],axis=-1)
        bbb = tf.transpose(aaa,[1,0])
        ccc = tf.one_hot(bbb,depth=300)
        nnn = tf.reshape(ccc,[-1,1200])
        res = sess.run(ccc)
        print(res)


    #input_layer


