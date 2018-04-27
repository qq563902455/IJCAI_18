# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from aeTools.autoEncoder import autoEncoder
from sklearn.metrics import log_loss
import lightgbm as lgb

pretrain = pd.read_csv('./ProcessedData/pretrain.csv')
valid = pd.read_csv('./ProcessedData/valid.csv')
train = pd.read_csv('./ProcessedData/train.csv')
test = pd.read_csv('./ProcessedData/test.csv')


pretrainX = pretrain.drop(['is_trade'], axis=1)
pretrainY = pretrain.is_trade

validX = valid.drop(['is_trade'], axis=1)
validY = valid.is_trade

trainX = train.drop(['is_trade'], axis=1)
trainY = train.is_trade

hide_size = 200
input_size = 95

model = LogisticRegression(C=1.0)
model = lgb.LGBMClassifier(
    random_state=666,
    max_depth=4,
    subsample=0.80,
    n_estimators=100,
    colsample_bytree=0.6,
#    reg_alpha=0.01,
    learning_rate=0.1,
#    reg_lambda=0.01,
    # is_unbalance=True,
    # scale_pos_weight=1,
    min_child_samples=15,
    subsample_freq=2,
)
ae = autoEncoder(hide_size=hide_size, input_size=input_size,
                 sparsity=8.0, gamma=0, p=0.2)

#def getOnehotValue(data):
#    result = np.zeros((data.shape[0], 2))
#    for i in range(data.shape[0]):
#        result[i, int(data[i])] = 1
#    return result


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    '''
    训练ae
    '''
    ae.train(epoch=5, batch_size=50, X=pretrainX.values,
             sess=sess, testX=validX.values)
    ae.train(epoch=5, batch_size=50, X=validX.values,
             sess=sess, testX=validX.values)


    x_train_representation = ae.encoder(pretrainX.values, sess)
    x_valid_representation = ae.encoder(validX.values, sess)
    
model.fit(x_train_representation, pretrainY)
print(log_loss(validY, model.predict_proba(x_valid_representation)[:, 1]))


