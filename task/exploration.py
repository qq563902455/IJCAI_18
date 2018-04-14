import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from lxyTools.singleModelUtils import singleModel
from sklearn.metrics import log_loss
import lightgbm as lgb
from lxyTools.featureSelect import maximalInformationCoef
from lxyTools.featureSelect import micCompute

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


minTime = pd.to_datetime('2018-9-17 16:00:00')
for i in range(8):
    trainData.loc[
        trainData.context_timestamp >= minTime+pd.to_timedelta(str(i)+' days'),
        'day'] += 1
    testData.loc[
        testData.context_timestamp >= minTime+pd.to_timedelta(str(i)+' days'),
        'day'] += 1

num1 = trainData[trainData.is_trade == 1].is_trade.count()
num0 = trainData[trainData.is_trade == 0].is_trade.count()

print('-----'*5)
print('为1的比例: ', num1/(num1+num0))
print('-----'*5)
print('train maxTime: ', trainData['context_timestamp'].max())
print('train minTime: ', trainData['context_timestamp'].min())
print('test maxTime: ', testData['context_timestamp'].min())
print('test minTime: ', testData['context_timestamp'].max())

print('-----'*5)
print('trainData cat:')
for col in trainData.columns:
    if 'id' in col:
        print(col, '\tnum: ', len(trainData[trainData.day==6][col].unique()))
print('-----'*5)
print('testData cat:')
for col in testData.columns:
    if 'id' in col:
        print(col, '\tnum: ', len(testData[col].unique()))

print('-----'*5)
print('rData cat:')
rData = trainData[trainData.day==6].sample(frac=0.33)
for col in trainData.columns:
    if 'id' in col:
        print(col, '\tnum: ', len(rData[col].unique()))

print('对比上面的这三个结果， 我们可以推断出测试集是由于最后一天的随机采样得到')
print('-----'*5)

alldata = trainData.append(testData)

plt.figure()
sns.countplot(alldata.day)

g = sns.FacetGrid(alldata, col="day")
g.map(sns.distplot, 'hour')
g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'hour', 'is_trade')

g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'user_gender_id', 'is_trade')

g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'user_occupation_id', 'is_trade')

sns.factorplot(x='context_page_id', y='is_trade', col='day', data=trainData)
sns.factorplot(x='context_page_id', y='is_trade', data=trainData)

plt.figure()
sns.barplot(x='day', y='is_trade', data=trainData)

plt.figure()
sns.countplot(alldata.hour)

plt.figure(figsize=(10,5))
sns.barplot(x="hour", y="is_trade", data=trainData);


featuresUsed = ['item_price_level', 'item_sales_level', 'item_collected_level',
                'item_pv_level', 'user_age_level', 'user_star_level',
                'shop_review_num_level', 'shop_review_positive_rate',
                'shop_star_level', 'shop_score_service', 'shop_score_delivery',
                'shop_score_description', 'is_trade']
corrmat = trainData[featuresUsed].corr()
plt.figure()
sns.heatmap(corrmat)

def getStringVal(s, num):
    for i in range(num-1):
        if ';' in s:
            pos = s.index(';')
            s = s[pos+1:]
        else:
            return -1
    if ';' in s:
        pos = s.index(';')
        s = s[:pos]
    return s


#trainData['item_category_list'].apply(lambda x: getStringVal(x,1)).unique()

#trainData['item_category_list'].apply(lambda x: getStringVal(x, 2)).unique()

#trainData['item_category_list'].head()


trainData['item_category_list'].apply(lambda x: getStringVal(x, 1)).head()

trainData['item_cat_id'] = trainData['item_category_list'].apply(lambda x: getStringVal(x, 2))
trainData['item_cat_len'] = trainData['item_category_list'].apply(lambda x: x.count(';'))
trainData['item_cat2_id'] = trainData['item_category_list'].apply(lambda x: getStringVal(x, 3))

trainData['item_cat2_id'].unique()
trainData['item_cat_id'] .unique()

plt.figure()
sns.countplot(trainData.item_cat_len)

plt.figure()
g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'item_cat_len', 'is_trade')

plt.figure()
g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'item_cat_id', 'is_trade')

plt.figure()
g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'item_cat2_id', 'is_trade')


trainData['item_property_len']=trainData['item_property_list'].apply(lambda x: x.count(';'))
sns.countplot(trainData['item_property_list'].apply(lambda x: x.count(';')))

plt.figure()
g = sns.FacetGrid(trainData, col="day")
g.map(sns.kdeplot, 'item_property_len')

sns.regplot(x="item_property_len", y="is_trade", data=trainData, x_estimator=np.mean)


#property_list = []
#
#def checkIn(x):
#    if x not in property_list:
#        property_list.append(x)
#        
#def countUniqueProperty(s):
#    totalNum = s.count(';')
#    for i in range(totalNum):
#        pos = s.index(';')
#        checkIn(s[:pos])
#        s = s[pos+1:]
#    checkIn(s)
#    return 0
#
#trainData['item_property_list'].apply(countUniqueProperty)
#len(property_list)
