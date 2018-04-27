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
    './rawdata/round1_ijcai_18_test_b_20180418.txt', sep=' ').instance_id

outId = pd.read_csv('./ProcessedData/day7.csv').instance_id

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

def metric(y_true, y_re):
    return log_loss(y_true, y_re)


model = lgb.LGBMClassifier(
    random_state=666,
    max_depth=4,
    subsample=0.8,
    n_estimators=1650,
    colsample_bytree=0.8,
#    reg_alpha=0.01,
    learning_rate=0.01,
#    reg_lambda=0.01,
    # is_unbalance=True,
    # scale_pos_weight=1,
    min_child_samples=5,
    subsample_freq=15,
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
#model = LogisticRegression(C=100)
#model.fit(pretrainX, pretrainY, eval_set=[(validX, validY)],
#          eval_metric='logloss', early_stopping_rounds=300)
#print(gc.collect())
#model = LogisticRegression(penalty='l1')
smodel = singleModel(model, kfold=StratifiedKFold(n_splits=5,
                                                  random_state=945,
                                                  shuffle=True))
smodel.fit(pretrainX, pretrainY, metric)
print(log_loss(validY, smodel.predict_proba(validX)[:, 1]))
print(gc.collect())


featureImportancelist = []
for m in smodel.modelList:
    featureImportancelist.append(m.feature_importances_)
importanceSeries = pd.Series(featureImportancelist[0])
for i in range(1, len(featureImportancelist)):
    importanceSeries += pd.Series(featureImportancelist[i])
importanceSeries.index = pretrainX.columns



#score_dict = {}
#for col in pretrainX.columns:
#    print('-'*20)
#    print(col)
#    smodel.fit(pretrainX.drop([col],axis=1), pretrainY, metric)
#    score = log_loss(validY, smodel.predict_proba(validX.drop([col],axis=1))[:, 1])
#    score_dict[col] = score
#    print(score)




# =============================================================================
# 特征筛选
# =============================================================================
#model = lgb.LGBMClassifier(
#    random_state=666,
#    max_depth=4,
#    subsample=0.8,
#    n_estimators=1650,
#    colsample_bytree=0.6,
##    reg_alpha=0.01,
#    learning_rate=0.01,
##    reg_lambda=0.01,
#    # is_unbalance=True,
#    # scale_pos_weight=1,
#    min_child_samples=15,
#    subsample_freq=2,
#)
#smodel1 = singleModel(model, kfold=StratifiedKFold(n_splits=5,
#                                                   random_state=945,
#                                                   shuffle=True))
#smodel1.fit(pretrainX, pretrainY, metric)
#print(log_loss(validY, smodel1.predict_proba(validX)[:, 1]))
#
#model = xgb.XGBClassifier(
#            max_depth=3,
#            n_estimators=300,
#            subsample=0.8,
#            colsample_bytree=0.6,
#            min_child_weight=2,
#            reg_alpha=0.3,
#            reg_lambda=0.7,
#            learning_rate=0.1,
#            n_jobs=4,
#        )
#smodel2 = singleModel(model, kfold=StratifiedKFold(n_splits=5,
#                                                   random_state=945,
#                                                   shuffle=True))
#smodel2.fit(pretrainX, pretrainY, metric)
#print(log_loss(validY, smodel2.predict_proba(validX)[:, 1]))
#
#
#featureImportancelist = []
#for m in smodel1.modelList:
#    featureImportancelist.append(m.feature_importances_)
#importanceSeries1 = pd.Series(featureImportancelist[0])
#for i in range(1, len(featureImportancelist)):
#    importanceSeries1 += pd.Series(featureImportancelist[i])
#importanceSeries1.index = pretrainX.columns
#
#featureImportancelist = []
#for m in smodel2.modelList:
#    featureImportancelist.append(m.feature_importances_)
#importanceSeries2 = pd.Series(featureImportancelist[0])
#for i in range(1, len(featureImportancelist)):
#    importanceSeries2 += pd.Series(featureImportancelist[i])
#importanceSeries2.index = pretrainX.columns
#
#
#importanceSeries = importanceSeries2.rank() + importanceSeries1.rank()
#
#numC = 48
#featureSelect = importanceSeries.sort_values().tail(numC).index
#
#
#
#model = lgb.LGBMClassifier(
#    random_state=666,
#    max_depth=4,
#    subsample=0.8,
#    n_estimators=100,
#    colsample_bytree=0.6,
##    reg_alpha=0.01,
#    learning_rate=0.1,
##    reg_lambda=0.01,
#    # is_unbalance=True,
#    # scale_pos_weight=1,
#    min_child_samples=15,
#    subsample_freq=2,
#)
#smodel = singleModel(model, kfold=StratifiedKFold(n_splits=5,
#                                                   random_state=945,
#                                                   shuffle=True))
#smodel.fit(pretrainX[featureSelect], pretrainY, metric)
#print(log_loss(validY, smodel.predict_proba(validX[featureSelect])[:, 1]))
#gc.collect()

# =============================================================================
# 融合本地测试
# =============================================================================
modellist=[
        lgb.LGBMClassifier(
            random_state=666,
            max_depth=4,
            subsample=0.80,
            n_estimators=1650,
            colsample_bytree=0.6,
            learning_rate=0.01,
            min_child_samples=15,
            subsample_freq=2),
        xgb.XGBClassifier(
            max_depth=3,
            n_estimators=2500,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=2,
            reg_alpha=0.3,
            reg_lambda=0.7,
            learning_rate=0.01,
            n_jobs=4,
        ),
        LogisticRegression(C=100)
        ]

stacker_model=stacker(modellist, higherModel=linearBlending([0,0,0],
                                                            30,
                                                            lambda x,y:-metric(x,y),
                                                            obj='classification'),
                      kFold=StratifiedKFold(n_splits=5,
                                            random_state=945,
                                            shuffle=True),
                      kFoldHigher=StratifiedKFold(n_splits=5,
                                                  random_state=777,
                                                  shuffle=True))
stacker_model.fit(pretrainX, pretrainY, metric)
print(log_loss(validY, stacker_model.predict_proba(validX)))
print(gc.collect())


for model in stacker_model.modelHigherList:
    print(model.paramList)


# =============================================================================
# 单模型
# =============================================================================
#model = lgb.LGBMClassifier(
#    random_state=666,
#    max_depth=4,
#    subsample=0.80,
#    n_estimators=1592,
#    colsample_bytree=0.6,
##    reg_alpha=0.01,
#    learning_rate=0.01,
##    reg_lambda=0.01,
#    # is_unbalance=True,
#    # scale_pos_weight=1,
#    min_child_samples=15,
#    subsample_freq=2
#)
#model = LogisticRegression(C=100)
#model=xgb.XGBClassifier(
#            max_depth=3,
#            n_estimators=150,
#            subsample=0.8,
#            colsample_bytree=0.6,
#            min_child_weight=2,
#            reg_alpha=0.3,
#            reg_lambda=0.7,
#            learning_rate=0.1,
#            n_jobs=4,
#        )
#smodel = singleModel(model, kfold=StratifiedKFold(n_splits=5,
#                                                  random_state=945,
#                                                  shuffle=True))
#
#smodel.fit(trainX, trainY, metric)
#out = smodel.predict_proba(test)[:,1]
#print(gc.collect())
    
    
# =============================================================================
#  stacking 融合模型
# =============================================================================
modellist=[
        lgb.LGBMClassifier(
            random_state=666,
            max_depth=4,
            subsample=0.80,
            n_estimators=1650,
            colsample_bytree=0.6,
            learning_rate=0.01,
            min_child_samples=15,
            subsample_freq=2),
#        xgb.XGBClassifier(
#            max_depth=3,
#            n_estimators=150,
#            subsample=0.8,
#            colsample_bytree=0.6,
#            min_child_weight=2,
#            reg_alpha=0.3,
#            reg_lambda=0.7,
#            learning_rate=0.1,
#            n_jobs=4,
#        ),
        LogisticRegression(C=100)
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
stacker_model.fit(trainX, trainY, metric)    
out = stacker_model.predict_proba(test)
print(gc.collect()) 
   

# =============================================================================
# 生成最后的结果
# =============================================================================
out = pd.DataFrame({'instance_id': outId,
                    'predicted_score': out})
out = out.set_index('instance_id')

out.loc[selectedOutId].to_csv('submit.txt', sep=' ')
print('end')
