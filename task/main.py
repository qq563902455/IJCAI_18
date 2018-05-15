import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from lxyTools.singleModelUtils import singleModel
from lxyTools.stacker import stacker
from lxyTools.stacker import linearBlending
from lxyTools.boostingTreeWithLM import BoosterLmClassifier
# from lxyTools.boostingTreeWithLM import FM
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import os
print(os.getpid())

# 读取数据
train = pd.read_csv('./ProcessedData/train.csv')
test = pd.read_csv('./ProcessedData/test.csv')


catFeatureslist = ['item_id', 'item_brand_id', 'item_city_id',
                   'user_id', 'user_gender_id', 'user_occupation_id',
                   'context_id', 'shop_id', 'item_cat_id',
                   ]
# 将类别特征转换成cat，这样lgb就可以直接识别出来
for col in train.columns:
    if col in catFeatureslist:
        train[col] = pd.Categorical(train[col])
        test[col] = pd.Categorical(test[col])

# 生成训练和输出用的数据
trainX = train.drop(['is_trade'], axis=1)
trainY = train.is_trade


testId = test.instance_id
test = test.drop(['instance_id'], axis=1)


# 获取最后需要输出的数据的id
print('read id')
selectedOutId = pd.read_table(
    './rawdata/' +
    'round2_ijcai_18_test_b_20180510.txt', sep=' ').instance_id
gc.collect()

# =============================================================================
# 用于计算lgb需要的树的数量
# =============================================================================
# 拆分验证集和训练集
X_train, X_test, y_train, y_test = train_test_split(
          trainX.drop(['item_id', 'shop_id',
                       'item_brand_id', 'item_city_id',
                       'user_id'], axis=1),
          trainY, test_size=0.2, random_state=1456)

model = lgb.LGBMClassifier(
    random_state=666,
    max_depth=6,
    subsample=0.8,
    n_estimators=10000,
    colsample_bytree=0.6,
    reg_alpha=0.3,
    learning_rate=0.05,
    reg_lambda=0.3,
    # is_unbalance=True,
    # scale_pos_weight=1,
    min_child_samples=150,
    subsample_freq=1,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          eval_metric='logloss', early_stopping_rounds=100)

del X_train
del X_test
del y_train
del y_test
gc.collect()


def metric(y_true, y_re):
    return log_loss(y_true, y_re)


# =============================================================================
# 单模型lgb
# =============================================================================
model = lgb.LGBMClassifier(
    random_state=666,
    max_depth=6,
    subsample=0.8,
    n_estimators=520,
    colsample_bytree=0.6,
    reg_alpha=0.3,
    learning_rate=0.05,
    reg_lambda=0.3,
    min_child_samples=150,
    subsample_freq=1,
)
smodel = singleModel(model, kfold=StratifiedKFold(n_splits=5,
                                                  random_state=945,
                                                  shuffle=True))
# 删除这些特征的原因是因为这些特征的类别太多了容易引发模型的过拟合
smodel.fit(trainX.drop(['item_id', 'shop_id',
                        'item_brand_id', 'item_city_id',
                        'user_id'], axis=1), trainY, metric)
print(gc.collect())
# 输出
out = smodel.predict_proba(test.drop(['item_id', 'shop_id',
                                      'item_brand_id', 'item_city_id',
                                      'user_id'], axis=1))[:, 1]

# =============================================================================
# lgb+lr 比赛中效果最好的模型，但是非常消耗内存
# 第一层包括5个lgb,每一个包含的特征都不相同
# 第二层为一个l2正则的lr
# =============================================================================
gc.collect()
# lgb模型
modellist = [
        # item_id tree
        lgb.LGBMClassifier(
                random_state=123,
                num_leaves=32,
                max_depth=6,
                n_estimators=20,
                learning_rate=0.5,
            ),
        # shop_id tree
        lgb.LGBMClassifier(
                random_state=123,
                num_leaves=20,
                max_depth=5,
                n_estimators=20,
                learning_rate=0.5,
            ),
        # item_brand_id tree
        lgb.LGBMClassifier(
                random_state=123,
                num_leaves=20,
                max_depth=5,
                n_estimators=20,
                learning_rate=0.5,
            ),
        # item_city_id tree
        lgb.LGBMClassifier(
                random_state=123,
                num_leaves=32,
                max_depth=6,
                n_estimators=20,
                learning_rate=0.5,
            ),
        # base tree
        lgb.LGBMClassifier(
                random_state=666,
                max_depth=6,
                subsample=0.8,
                n_estimators=250,
                colsample_bytree=0.6,
                reg_alpha=0.3,
                learning_rate=0.1,
                reg_lambda=0.3,
                min_child_samples=150,
                subsample_freq=1,
            ),

        ]


featurelist = [['item_id'], ['shop_id'], ['item_brand_id'], ['item_city_id']]
# featurelist = []
featurelist.append(list(trainX.drop(['item_id', 'shop_id',
                                     'item_brand_id', 'item_city_id',
                                     'user_id'], axis=1).columns))
# lr模型
lm = LogisticRegression(C=0.0004, penalty='l2')
# lgb+lr
lgblr = BoosterLmClassifier(modellist, featurelist, lm)
smodel = singleModel(lgblr, kfold=StratifiedKFold(n_splits=5,
                                                  random_state=945,
                                                  shuffle=True))
smodel.fit(trainX, trainY, metric)
gc.collect()
# 输出
out = smodel.predict_proba(test)[:, 1]


# =============================================================================
# stacking
# 这样一种方式的融合可以使得线下交叉验证效果变好
# 但是效果没有lgb+lr好
# =============================================================================
modellist = [
            lgblr,
            lgb.LGBMClassifier(
                random_state=666,
                max_depth=6,
                subsample=0.8,
                n_estimators=2800,
                colsample_bytree=0.6,
                reg_alpha=0.3,
                learning_rate=0.01,
                reg_lambda=0.3,
                # is_unbalance=True,
                # scale_pos_weight=1,
                min_child_samples=150,
                subsample_freq=1,
            ),
        ]

featurelist = []
featurelist.append(trainX.columns)
featurelist.append(trainX.drop(['item_id', 'shop_id',
                                'item_brand_id', 'item_city_id',
                                'user_id'], axis=1).columns)


stacker_model = stacker(modellist,
                        higherModel=linearBlending([0, 0],
                                                   30,
                                                   lambda x, y: -metric(x, y),
                                                   obj='classification'),
                        kFold=StratifiedKFold(n_splits=5,
                                              random_state=945,
                                              shuffle=True),
                        kFoldHigher=StratifiedKFold(n_splits=5,
                                                    random_state=945,
                                                    shuffle=True))
stacker_model.fit(trainX, trainY, metric, featuresList=featurelist)
out = stacker_model.predict_proba(test, featurelist)


# =============================================================================
# 生成最后的结果
# =============================================================================
out = pd.DataFrame({'instance_id': testId,
                    'predicted_score': out})
out = out.set_index('instance_id')

out.loc[selectedOutId].to_csv('./ProcessedData/submit.txt', sep=' ')
print(out.loc[selectedOutId].head())
print('end')
