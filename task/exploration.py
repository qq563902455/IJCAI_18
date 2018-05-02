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
import random
import gc
gc.collect()
random.seed(10)


flag=False
def rowRandomSelect(x):
    global flag
    if not flag:
        print('a')
        flag=True
        return False
    else:
        return random.random()>0.3


testData = pd.read_table(
    './rawdata/round2_ijcai_18_test_a_20180425.txt', sep=' ')
trainData = pd.read_table(
    './rawdata/round2_train.txt', sep=' ', skiprows=rowRandomSelect)

testData.head()

trainData.columns

trainData['context_timestamp'] = pd.to_datetime(
    trainData['context_timestamp'], unit='s')

testData['context_timestamp'] = pd.to_datetime(
    testData['context_timestamp'], unit='s')


trainData.context_timestamp += pd.to_timedelta('8:00:00')
testData.context_timestamp += pd.to_timedelta('8:00:00')

trainData.insert(loc=0, column='day', value=trainData.context_timestamp.dt.day)
trainData.insert(loc=0, column='hour', value=trainData.context_timestamp.dt.hour)
trainData.loc[trainData.day==31, 'day']=0

testData.insert(loc=0, column='day', value=testData.context_timestamp.dt.day)
testData.insert(loc=0, column='hour', value=testData.context_timestamp.dt.hour)


print('训练数据与测试数据的时间跨度')
print(trainData.context_timestamp.min(),'-',trainData.context_timestamp.max())
print(testData.context_timestamp.min(),'-',testData.context_timestamp.max())



#minTime = pd.to_datetime('2018-8-30 16:00:00')
#for i in range(8):
#    trainData.loc[
#        trainData.context_timestamp >= minTime+pd.to_timedelta(str(i)+' days'),
#        'day'] += 1
#    testData.loc[
#        testData.context_timestamp >= minTime+pd.to_timedelta(str(i)+' days'),
#        'day'] += 1

print('train day\t', trainData.day.unique())
print('test day\t', testData.day.unique())

num1 = trainData[trainData.is_trade == 1].is_trade.count()
num0 = trainData[trainData.is_trade == 0].is_trade.count()

print('-----'*5)
print('为1的比例: ', num1/(num1+num0))
print('-----'*5)



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
#print('-----'*5)
#print('rData cat:')
#rData = trainData[trainData.day==6].sample(frac=0.33)
#for col in trainData.columns:
#    if 'id' in col:
#        print(col, '\tnum: ', len(rData[col].unique()))
#
#print('对比上面的这两个结果， 我们可以推断出测试集A是由于最后一天的随机采样得到')
#print('-----'*5)

print('4-16这段时间里面样本数目与总体样本之间的关系')
for i in range(8):
    numP = trainData[np.array(trainData.day==i)&
                     np.array(trainData.hour<12)].instance_id.count()
    numSum = trainData[trainData.day==i].instance_id.count()
    numO = numSum - numP
    print('day',i,numP, '\t', numO, '\t', float(numO)/numP)

print(trainData[trainData.day==7].instance_id.count())


print(testData.instance_id.count())

alldata = trainData.append(testData)


plt.figure()
sns.countplot(trainData[trainData.day==5].hour)

plt.figure()
sns.countplot(trainData[trainData.day==6].hour)

plt.figure()
sns.countplot(trainData[trainData.day==7].hour)

plt.figure()
sns.countplot(testData[testData.day==7].hour)



print('alldata cat:')
print('-----'*5,'all')
for col in trainData.columns:
    if 'id' in col:
        print(col, '\tnum: ', len(alldata[col].unique()))
    
#for col in trainData.columns:
#    if 'id' in col:
#        plt.figure()
#        sns.barplot(x='day',y=col, data=alldata, estimator=lambda x:len(np.unique(x)))

plt.figure()
sns.countplot(trainData.hour)

plt.figure()
sns.countplot(trainData.day)

g = sns.FacetGrid(alldata, col="day")
g.map(sns.countplot, 'hour')

g = sns.FacetGrid(alldata, col="day")
g.map(sns.countplot, 'hour')
g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'hour', 'is_trade')

gc.collect()

g = sns.FacetGrid(trainData[trainData.day!=7], col="day")
g.map(sns.barplot, 'hour', 'is_trade')

g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'user_gender_id', 'is_trade')

g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'user_occupation_id', 'is_trade')

#sns.factorplot(x='context_page_id', y='is_trade', col='day', data=trainData)
#sns.factorplot(x='context_page_id', y='is_trade', data=trainData)
#
#plt.figure()
#sns.barplot(x='day', y='is_trade', data=trainData)
#
#plt.figure()
#sns.countplot(alldata.hour)
#
#plt.figure(figsize=(10,5))
#sns.barplot(x="hour", y="is_trade", data=trainData);


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

#trainData['item_cat2_id'].unique()
#trainData['item_cat_id'] .unique()

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

#for i in trainData.item_cat_id.unique():
#    print(trainData[trainData.item_cat_id==i].count())
