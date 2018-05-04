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

selectedOutId = pd.read_table(
    './rawdata/round2_ijcai_18_test_a_20180425.txt', sep=' ').instance_id

valid = pd.read_csv('./ProcessedData/valid.csv')
train = pd.read_csv('./ProcessedData/train.csv')
test = pd.read_csv('./ProcessedData/test.csv')


trainX = train.drop(['is_trade'], axis=1)
trainY = train.is_trade


validX = valid.drop(['is_trade'], axis=1)
validY = valid.is_trade


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
    n_estimators=2000,
    colsample_bytree=0.6,
    reg_alpha=0.3,
    learning_rate=0.01,
    reg_lambda=0.3,
    # is_unbalance=True,
    # scale_pos_weight=1,
    min_child_samples=150,
    subsample_freq=1,
)
#model = xgb.XGBClassifier(
#            max_depth=3,
#            n_estimators=2500,
#            subsample=0.8,
#            colsample_bytree=0.6,
#            min_child_weight=2,
#            reg_alpha=0.3,
#            reg_lambda=0.7,
#            learning_rate=0.01,
#            n_jobs=4,
#        )
#model = LogisticRegression(C=100000)
#model.fit(trainX, trainY, eval_set=[(validX, validY)],
#          eval_metric='logloss', early_stopping_rounds=300)
#print(gc.collect())
#model = LogisticRegression(penalty='l1')
smodel = singleModel(model, kfold=StratifiedKFold(n_splits=5,
                                                  random_state=945,
                                                  shuffle=True))
smodel.fit(validX, validY, metric)
print(gc.collect())

featureImportancelist = []
for m in smodel.modelList:
    featureImportancelist.append(m.feature_importances_)
importanceSeries = pd.Series(featureImportancelist[0])
for i in range(1, len(featureImportancelist)):
    importanceSeries += pd.Series(featureImportancelist[i])
importanceSeries.index = validX.columns
#print(log_loss(validY, smodel.predict_proba(validX)[:, 1]))

#test = test.drop(['day'], axis=1)
out = smodel.predict_proba(test)
print(gc.collect()) 


validX = validX.drop(['user_id_timeperiod'], axis=1)


modellist=[
            lgb.LGBMClassifier(
                random_state=666,
                max_depth=6,
                subsample=0.8,
                n_estimators=1500,
                colsample_bytree=0.8,
                reg_alpha=0.10,
                learning_rate=0.01,
                reg_lambda=0.3,
                # is_unbalance=True,
                # scale_pos_weight=1,
                min_child_samples=150,
                subsample_freq=1,
            ),
#        xgb.XGBClassifier(
#            max_depth=3,
#            n_estimators=2500,
#            subsample=0.8,
#            colsample_bytree=0.6,
#            min_child_weight=2,
#            reg_alpha=0.3,
#            reg_lambda=0.7,
#            learning_rate=0.01,
#            n_jobs=4,
#        ),
        LogisticRegression(C=100000)
        ]

stacker_model=stacker(modellist, higherModel=linearBlending([0,0],
                                                            20,
                                                            lambda x,y:-metric(x,y),
                                                            obj='classification'),
                      kFold=StratifiedKFold(n_splits=5,
                                            random_state=945,
                                            shuffle=True),
                      kFoldHigher=StratifiedKFold(n_splits=5,
                                                  random_state=777,
                                                  shuffle=True))
stacker_model.fit(validX, validY, metric)
out = stacker_model.predict_proba(test)
out = pd.DataFrame({'instance_id': selectedOutId,
                    'predicted_score': out})
# =============================================================================
# 生成最后的结果
# =============================================================================
out = pd.DataFrame({'instance_id': selectedOutId,
                    'predicted_score': out[:,1]})

    
#out.loc[out.predicted_score>1, 'predicted_score']=1

out.to_csv('submit.txt', sep=' ', index=False)
print('end')
