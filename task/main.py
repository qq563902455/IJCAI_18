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
import gc




datalist = []
for i in range(2, 8):
    datalist.append(pd.read_csv('./ProcessedData/day'+str(i)+'.csv'))

selectedOutId = pd.read_table(
    './rawdata/round1_ijcai_18_test_b_20180418.txt', sep=' ').instance_id

outId = datalist[5].instance_id

allData = pd.concat(datalist)

catFeatureslist = ['item_id', 'item_brand_id', 'item_city_id',
                   'user_id', 'user_gender_id', 'user_occupation_id',
                   'context_id', 'shop_id',
                  ]

catDropList = ['item_id', 'item_brand_id', 'item_city_id',
               'user_id', 'user_gender_id', 'user_occupation_id',
               'context_id', 'shop_id', 
               'hour'
               ]

allData = allData.drop(
    ['context_timestamp',
     'item_category_list', 'instance_id',
     'item_property_list', 'predict_category_property'], axis=1)


allData = allData.drop(catDropList, axis=1)

for col in allData.columns:
    if col in catFeatureslist:
        allData[col] = pd.Categorical(allData[col])

for col in allData:
    if col not in catFeatureslist:
        if col not in ['day', 'is_trade']:
            minval = allData[col].min()
            maxval = allData[col].max()
            if maxval != minval:
                allData[col] = allData[col].apply(
                    lambda x: (x-minval)/(maxval-minval))
            else:
                allData = allData.drop([col], axis=1)
                

pretrain = allData[allData.day <= 5].drop(['day'], axis=1)
pretrainX = pretrain.drop(['is_trade'], axis=1)
pretrainY = pretrain.is_trade

valid = allData[allData.day == 6].drop(['day'], axis=1)
validX = valid.drop(['is_trade'], axis=1)
validY = valid.is_trade

train = allData[allData.day <= 6]
train = train[train.day > 2].drop(['day'], axis=1)
trainX = train.drop(['is_trade'], axis=1)
trainY = train.is_trade

test = allData[allData.day == 7][trainX.columns]

del allData
del pretrain
del valid

gc.collect()


def metric(y_true, y_re):
    return log_loss(y_true, y_re)


#model = lgb.LGBMClassifier(
#    random_state=666,
#    max_depth=4,
#    subsample=0.80,
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
model = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=2,
            reg_alpha=0.3,
            reg_lambda=0.7,
            learning_rate=0.1,
            n_jobs=4,
        )
#model = LogisticRegression(C=100)
#model.fit(pretrainX, pretrainY, eval_set=[(validX, validY)],
#          eval_metric='logloss', early_stopping_rounds=30)
#print(gc.collect())

smodel = singleModel(model, kfold=StratifiedKFold(n_splits=5,
                                                  random_state=945,
                                                  shuffle=True))
smodel.fit(pretrainX, pretrainY, metric)
print(log_loss(validY, smodel.predict_proba(validX)[:, 1]))
print(gc.collect())
#validPre1 = smodel.predict_proba(validX)[:, 1]


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

modellist=[
#        lgb.LGBMClassifier(
#            random_state=666,
#            max_depth=4,
#            subsample=0.80,
#            n_estimators=1592,
#            colsample_bytree=0.6,
#            learning_rate=0.01,
#            min_child_samples=15,
#            subsample_freq=2),
        xgb.XGBClassifier(
            max_depth=3,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=2,
            reg_alpha=0.3,
            reg_lambda=0.7,
            learning_rate=0.1,
            n_jobs=4,
        ),
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
            n_estimators=1592,
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
