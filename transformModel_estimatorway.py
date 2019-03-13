

import time
import tensorflow as tf

from lib.read_conf import Config
import datasets
import make_column_feature
import testvars4
from build_estimator import build_dnn_estimator
import make_column_feature
import argparse




#model_dir = './model_dir'
#export_dir = './export_dir'
parser = argparse.ArgumentParser(description='Train DNN Model.')
CONFIG = Config()

##model para
parser.add_argument(
    '--model_dir',
    type=str,
    default=CONFIG.train["model_dir"],
    help='Base directory for the model.')

parser.add_argument(
    '--train_epochs',
    type=int,
    default=CONFIG.train["train_epochs"],
    help='Number of training epochs.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=CONFIG.train["batch_size"],
    help='Number of examples per batch.')
parser.add_argument(
    '--train_data',
    type=str,
    default=CONFIG.train["train_data"],
    help='Path to the train data.')
parser.add_argument(
    '--eval_data',
    type=str,
    default=CONFIG.train["eval_data"],
    help='Path to the validation data.')
parser.add_argument(
    '--profiler_save_steps',
    type=int,
    default=CONFIG.train['profiler_save_steps'],
    help='Save steps for profiler monitoring.')

parser.add_argument(
    '--model_type',
    type=str,
    default='dnn',
    help='Base selection for the model.')


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



def train_and_eval(model):
    for n in range(FLAGS.train_epochs):
        tf.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        train_data = FLAGS.train_data
        tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, train_data))

################################ observe the weight of specified item ###############################################
        #tensors_to_log = {
        #    'cross/cross_1/weight/(weight)'   ####get names from tensorboard graph
        #}
        #logging_hook = [tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)]
#####################################################################################################################
        phhooks = [
            tf.train.ProfilerHook(
                output_dir=CONFIG.train["model_timeline_dir"],
                save_steps=FLAGS.profiler_save_steps,
                show_memory=False)
        ]
        model.train(
            input_fn=
            lambda: datasets.input_fn(FLAGS.train_data, 'train', FLAGS.batch_size),
            hooks=phhooks,
            steps=None,
            max_steps=None,
            saving_listeners=None)

        print('-' * 80)
        tf.logging.info('<EPOCH {}>: Start evaluating {}'.format(n + 1, FLAGS.eval_data))

        with tf.contrib.tfprof.ProfileContext(CONFIG.train["evaluate_timeline_dir"]) as pctx:
            results = model.evaluate(
                input_fn=lambda: datasets.input_fn(FLAGS.eval_data, 'eval', FLAGS.batch_size),
                steps=None,
                hooks=None,
                checkpoint_path=None,
                name=None)

        print('-' * 80)
        # Display evaluation metrics
        for key in sorted(results):
            print('{}: {}'.format(key, results[key]))

        #save code
        con, xgbcon, cat = make_column_feature.transform_feature()
        feature_spec = tf.feature_column.make_parse_example_spec(set(con + xgbcon + cat))
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec)
        export_dir = model.export_savedmodel(  # export tf serving model
            model_dir,
            serving_input_receiver_fn)
        print("exported serving dir: {}".format(model_dir))







if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    CONFIG = Config()
    print("Using TensorFlow Version %s" % tf.__version__)
    model_dir = FLAGS.model_dir
    model_type = FLAGS.model_type
    print('\nModel Directory: {}'.format(model_dir))

    model = build_dnn_estimator(model_dir,model_type)
    #    model = build_custom_estimator(model_dir, FLAGS.model_type)
    tf.logging.info('Build estimator: {}'.format(model))

    train_fn = train_and_eval

    train_fn(model)



    #data = datasets.input_fn("test_data1.csv",mode="train",batch_size=10)
    #print(data)

    #conFeatureCol, FeatureCol_forxgb, catFeatureCol = make_column_feature.transform_feature()
    #print(conFeatureCol)
    #ouputType = [tf.int32]*4

    #featureTensors = transform2cat(tensor)

    #transform_XGB_Model = autograph.to_graph(testvars4.XGBprocess4)
    #print(autograph.to_code(testvars4.XGBprocess))
###################################################################################
# after add xgb code into estimator
    # model = build_dnn_estimator(FLAGS.model_dir,'dnn')
    # model.train(
    #     input_fn=
    #     lambda: datasets.input_fn("test_data1.csv",mode="train",batch_size=10),
    #     hooks=None,
    #     steps=None,
    #     max_steps=None,
    #    saving_listeners=None)



#save code
    # con,xgbcon,cat= make_column_feature.transform_feature()
    # feature_spec = tf.feature_column.make_parse_example_spec(set(con+xgbcon+cat) )
    # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    #     feature_spec)
    # export_dir = model.export_savedmodel(  # export tf serving model
    #     'export\dcn_output\{1}',
    #     serving_input_receiver_fn)
    # print("exported serving dir: {}".format(export_dir))
###################################################################################


    # with tf.Session() as sess:
    #     coninputTensor = tf.feature_column.input_layer(data, conFeatureCol)
    #     xgbinputTensor = tf.feature_column.input_layer(data, FeatureCol_forxgb)
    #     catinputTensor = tf.feature_column.input_layer(data, conFeatureCol)
    #     # print(xgbinputTensor.shape)
    #     # xgbinputTensor =  tf.transpose(xgbinputTensor,perm=[1,0])
    #     # print(xgbinputTensor.shape)
    #     xgbFeature = tf.map_fn(transform_XGB_Model,xgbinputTensor,dtype=ouputType)
    #     #xgbFeature = transform_XGB_Model(xgbinputTensor)
    #     #dnnFeature = tf.concat([coninputTensor,catinputTensor,xgbFeature],axis=-1)
    #     aaa = tf.concat([xgbFeature],axis=-1)
    #     bbb = tf.transpose(aaa,[1,0])
    #     ccc = tf.one_hot(bbb,depth=300)
    #     res = sess.run(ccc)
    #     print(res)


    #input_layer