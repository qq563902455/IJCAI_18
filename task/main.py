import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from lxyTools.singleModelUtils import singleModel
from lxyTools.featureSelect import rfeBySingleModel
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
import gc




datalist = []
for i in range(2, 8):
    datalist.append(pd.read_csv('./ProcessedData/day'+str(i)+'.csv'))

temp_valid = pd.read_csv('./ProcessedData/temp_valid.csv')



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
temp_valid = temp_valid[allData.columns]

for col in allData.columns:
    if col in catFeatureslist:
        allData[col] = pd.Categorical(allData[col])
        temp_valid[col] = pd.Categorical(temp_valid[col])

for col in allData:
    if col not in catFeatureslist:
        if col not in ['day', 'is_trade']:
            minval = allData[col].min()
            maxval = allData[col].max()
            if maxval != minval:
                allData[col] = allData[col].apply(
                    lambda x: (x-minval)/(maxval-minval))
                temp_valid[col] = temp_valid[col].apply(
                    lambda x: (x-minval)/(maxval-minval))
            else:
                temp_valid = temp_valid.drop([col], axis=1)
                allData = allData.drop([col], axis=1)
                
temp_valid = temp_valid.drop(['day'], axis=1)
temp_validX = temp_valid.drop(['is_trade'], axis=1)
temp_validY = temp_valid.is_trade

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
    min_child_samples=40,
    subsample_freq=2,
)
#model = LogisticRegression()
#model.fit(pretrainX, pretrainY, eval_set=(validX, validY),
#          eval_metric='logloss', early_stopping_rounds=100)
smodel = singleModel(model, kfold=StratifiedKFold(n_splits=5,
                                                  random_state=945,
                                                  shuffle=True))
smodel.fit(pretrainX, pretrainY, metric)
print(log_loss(validY, smodel.predict_proba(validX)[:, 1]))

print(log_loss(temp_validY, smodel.predict_proba(temp_validX)[:, 1]))
#validPre1 = smodel.predict_proba(validX)[:, 1]


featureImportancelist = []
for m in smodel.modelList:
    featureImportancelist.append(m.feature_importances_)
importanceSeries = pd.Series(featureImportancelist[0])
for i in range(1, len(featureImportancelist)):
    importanceSeries += pd.Series(featureImportancelist[i])
importanceSeries.index = pretrainX.columns

#scoreRe = rfeBySingleModel(
#        smodel, step=1, objnum=10, X=pretrainX, y=pretrainY,
#        valid=(validX, validY), metric=metric)
#
#featureUsed = scoreRe[1]


#score_dict = {}
#for col in pretrainX.columns:
#    print('-'*20)
#    print(col)
#    smodel.fit(pretrainX.drop([col],axis=1), pretrainY, metric)
#    score = log_loss(validY, smodel.predict_proba(validX.drop([col],axis=1))[:, 1])
#    score_dict[col] = score
#    print(score)




#model = lgb.LGBMClassifier(
#    random_state=666,
#    max_depth=4,
#    subsample=0.80,
#    n_estimators=1500,
#    colsample_bytree=0.6,
##    reg_alpha=0.01,
#    learning_rate=0.01,
##    reg_lambda=0.01,
#    # is_unbalance=True,
#    # scale_pos_weight=1,
#    min_child_samples=40,
#    subsample_freq=2
#)
#smodel = singleModel(model, kfold=StratifiedKFold(n_splits=5,
#                                                  random_state=222,
#                                                  shuffle=True))
#
#smodel.fit(trainX, trainY, metric)
#
#out = smodel.predict_proba(test)[:,1]
#out = pd.DataFrame({'instance_id': outId,
#                    'predicted_score': out})
#out.to_csv('submit.txt', sep=' ', index=False)
#print('end')
