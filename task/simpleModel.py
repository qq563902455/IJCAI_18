import pandas as pd
# from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import StratifiedKFold
#from lxyTools.singleModelUtils import singleModel
from sklearn.metrics import log_loss
import lightgbm as lgb


def metric(y_true, y_re):
    return log_loss(y_true, y_re)


testData = pd.read_table(
    './rawdata/round1_ijcai_18_test_a_20180301.txt', sep=' ')
trainData = pd.read_table(
    './rawdata/round1_ijcai_18_train_20180301.txt', sep=' ')

trainData['context_timestamp'] = pd.to_datetime(
    trainData['context_timestamp'], unit='s')

testData['context_timestamp'] = pd.to_datetime(
    testData['context_timestamp'], unit='s')


trainData.insert(loc=0, column='day', value=-1)
trainData.insert(loc=0, column='hour', value=trainData.context_timestamp.dt.hour)

testData.insert(loc=0, column='day', value=-1)
testData.insert(loc=0, column='hour', value=testData.context_timestamp.dt.hour)

#featuresUsed = ['item_price_level', 'item_sales_level', 'item_collected_level',
#                'item_pv_level', 'user_age_level', 'user_star_level',
#                'shop_review_num_level', 'shop_review_positive_rate',
#                'shop_star_level', 'shop_score_service', 'shop_score_delivery',
#                'shop_score_description']

featuresUsed = ['shop_score_delivery', 'shop_score_service', 'shop_score_description',
                'item_sales_level', 'item_price_level', 'user_occupation_id',
                'item_city_id', 'item_collected_level', 'item_pv_level',
                'user_age_level', 'shop_id']
catAllList = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id',
              'user_id', 'user_gender_id', 'user_occupation_id',
              'context_id', 'shop_id']

catUsedlist = []
for col in featuresUsed:
    if col in catAllList:
        catUsedlist.append(col)


validData = trainData[trainData.context_timestamp > '2018-9-23 16:00:00']
validData = validData[validData.context_timestamp <= '2018-9-24 16:00:00']
trainData = trainData[trainData.context_timestamp <= '2018-9-23 16:00:00']

trainX = trainData[featuresUsed]
trainY = trainData.is_trade

validX = validData[featuresUsed]
validY = validData.is_trade


model = lgb.LGBMClassifier(
    random_state=666,
    max_depth=3,
    subsample=0.8,
    n_estimators=1000,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    # reg_lambda=0.01,
    min_child_samples=40,
    # num_leaves=80,
    # subsample_freq=2
)

model.fit(trainX, trainY, eval_set=(validX, validY), eval_metric='logloss',
           early_stopping_rounds=100, categorical_feature=catUsedlist)

#out = model.predict_proba(testX)[:, 1]
#out = pd.DataFrame({'instance_id': testData['instance_id'],
#                    'predicted_score': out})
#out.to_csv('submit.txt', sep=' ', index=False)



