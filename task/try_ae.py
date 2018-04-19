# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

pretrain = pd.read_csv('./ProcessedData/pretrain.csv')
valid = pd.read_csv('./ProcessedData/valid.csv')
train = pd.read_csv('./ProcessedData/train.csv')
test = pd.read_csv('./ProcessedData/test.csv')


pretrainX = pretrain.drop(['is_trade'], axis=1)
pretrainY = pretrain.is_trade

validX = valid.drop(['is_trade'], axis=1)
validY = valid.is_trade

train = train[train.day > 2].drop(['day'], axis=1)
trainX = train.drop(['is_trade'], axis=1)
trainY = train.is_trade