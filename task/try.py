import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from lxyTools.singleModelUtils import singleModel
from sklearn.metrics import log_loss
import lightgbm as lgb


def metric(y_true, y_re):
    return log_loss(y_true, y_re)

testData = pd.read_table(
    './rawdata/round1_ijcai_18_test_a_20180301.txt', sep=' ')
trainData = pd.read_table(
    './rawdata/round1_ijcai_18_train_20180301.txt', sep=' ')
trainData.count()
testData.count()
trainData['context_timestamp'] = pd.to_datetime(
    trainData['context_timestamp'], unit='s')

testData['context_timestamp'] = pd.to_datetime(
    testData['context_timestamp'], unit='s')

trainData = trainData[trainData.context_timestamp > '2018-9-20 16:00:00']

featuresUsed = ['item_price_level', 'item_sales_level', 'item_collected_level',
                'item_pv_level', 'user_age_level', 'user_star_level',
                'shop_review_num_level', 'shop_review_positive_rate',
                'shop_star_level', 'shop_score_service', 'shop_score_delivery',
                'shop_score_description']

trainY = trainData['is_trade']
trainX = trainData[featuresUsed]
testX = testData[featuresUsed]

for col in trainX.columns:
    minval = min(trainX[col].min(), testX[col].min())
    maxval = max(trainX[col].max(), testX[col].max())
    trainX[col] = trainX[col].apply(lambda x: (x-minval)/(maxval-minval))
    testX[col] = testX[col].apply(lambda x: (x-minval)/(maxval-minval))


model = lgb.LGBMClassifier(
    random_state=666,
    max_depth=6,
    subsample=0.8,
    n_estimators=100,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    # reg_lambda=0.01,
    min_child_samples=40,
    # num_leaves=80,
    # subsample_freq=2
)
smodel = singleModel(model, kfold=StratifiedKFold(n_splits=5,
                                                  random_state=945,
                                                  shuffle=True))
smodel.fit(trainX, trainY, metric)

out = smodel.predict_proba(testX)[:, 1]
out = pd.DataFrame({'instance_id': testData['instance_id'],
                    'predicted_score': out})
out.to_csv('submit.txt', sep=' ', index=False)
