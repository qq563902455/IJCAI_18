import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from lxyTools.singleModelUtils import singleModel
from lxyTools.stacker import stacker
from lxyTools.stacker import linearBlending
from lxyTools.featureSelect import rfeBySingleModel
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
import gc
import np.numpy
from sklearn.preprocessing import OneHotEncoder

selectedOutId = pd.read_table(
    './rawdata/round2_ijcai_18_test_a_20180425.txt', sep=' ').instance_id
        

train = pd.read_csv('./ProcessedData/train.csv')
test = pd.read_csv('./ProcessedData/test.csv')


trainX = train.drop(['is_trade'], axis=1)
trainY = train.is_trade


testId = test.instance_id
test = test.drop(['instance_id'], axis=1)

#
#droplist = []
#for col in trainX.columns:
#    if '-2' in col or '-1' in col:
#        droplist.append(col)
#
#trainX = trainX.drop(droplist, axis=1)
#test = test.drop(droplist, axis=1)


def metric(y_true, y_re):
    return log_loss(y_true, y_re)

model = lgb.LGBMClassifier(
    random_state=666,
    max_depth=6,
    subsample=0.8,
    n_estimators=10,
    colsample_bytree=0.6,
    reg_alpha=0.3,
    learning_rate=1,
    reg_lambda=0.3,
    # is_unbalance=True,
    # scale_pos_weight=1,
    min_child_samples=150,
    subsample_freq=1,
)
model.fit(trainX, trainY)

temp = model.booster_
out = temp.predict(trainX ,pred_leaf=True)


#dsum = 0
#for i in range(10):
#    dsum += len(np.unique(out[:,i]))


one = OneHotEncoder(dtype=np.uint8)
one.fit(out)
onhot_out = one.transform(out).toarray()

lmodel = LogisticRegression()

lmodel.fit(onhot_out, trainY)


gc.collect()
