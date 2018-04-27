import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
# %matplotlib inline

testData = pd.read_table(
    './rawdata/round1_ijcai_18_test_a_20180301.txt', sep=' ')
testData_b = pd.read_table(
    './rawdata/round1_ijcai_18_test_b_20180418.txt', sep=' ')
trainData = pd.read_table(
    './rawdata/round1_ijcai_18_train_20180301.txt', sep=' ')

testData = pd.concat([testData, testData_b])


allData = trainData.append(testData)
del testData
del trainData

allData['context_timestamp'] = pd.to_datetime(
    allData['context_timestamp'], unit='s')


allData.insert(loc=0, column='day', value=-1)
allData.insert(loc=0, column='hour', value=allData.context_timestamp.dt.hour)


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


allData.insert(loc=0, column='item_cat_id',
               value=allData['item_category_list'].apply(lambda x: getStringVal(x, 2)))
allData.insert(loc=0, column='item_cat_len',
               value=allData['item_category_list'].apply(lambda x: x.count(';')))


minTime = pd.to_datetime('2018-9-17 16:00:00')
for i in range(8):
    allData.loc[
        allData.context_timestamp >= minTime+pd.to_timedelta(str(i)+' days'),
        'day'] += 1



    

def getColInfo(col, dataset, rate_col=False, nameAdd=''):
    name = ''
    for i in col:
        name += i+'_'
    name += nameAdd

    table_info = dataset[col+['day']].groupby(by=col,
                                              as_index=False).count()
    if not rate_col:
        table_info = table_info.fillna(0)
        table_info = table_info.rename(columns={'day': name+'counts'})
    else:
        table_1_info = dataset[dataset['is_trade'] == 1][
            col+['day']].groupby(by=col, as_index=False).count()
        table_info = pd.merge(left=table_info, right=table_1_info, how='outer',
                              on=col)
        table_info = table_info.fillna(0)
        table_info.insert(loc=0, column=name+'rate', value=0)
        table_info.loc[table_info['day_y'] == 0, name+'rate'] = 0
        table_info.loc[table_info['day_y'] != 0, name+'rate'] = \
            table_info.loc[table_info['day_y'] != 0, 'day_y']\
            / table_info.loc[table_info['day_y'] != 0, 'day_x']

        table_info = table_info.drop(['day_y'], axis=1)
        # table_info = table_info.drop(['day_x'], axis=1)
        table_info = table_info.rename(columns={'day_x': name+'counts'})
        del table_1_info
    gc.collect()
    return table_info


#def getColMeanInfo(col, dataset, meanCol, nameAdd=''):
#    name = ''
#    for i in col:
#        name += i+'_'
#    name += meanCol
#    
#    name += nameAdd
#    table_info = dataset[col+[meanCol]].groupby(by=col,
#                                              as_index=False).mean()
#    
#    table_info = table_info.fillna(0)
#    table_info = table_info.rename(columns={meanCol: name+'_mean'})
#
#    gc.collect()
#    return table_info


def mergeInfo(rawData, datasetlist):
    for dataset in datasetlist:
        index_cols = []
        for col in rawData.columns:
            if col in dataset.columns:
                index_cols.append(col)

        rawData = pd.merge(left=rawData, right=dataset,
                           how='left', on=index_cols)
    rawData = rawData.fillna(0)
    return rawData

def mergeIdRankInfo(dataset, collist):
    for col in collist:
        if type(col) == str:
            print('processing:\t', col)
            idlist = list(dataset[col].unique())
            dataset[col+'_rank'] = 0
            dataset[col+'_timeperiod'] = 0
            for i in tqdm(range(len(idlist))):
                countNum = dataset[dataset[col]==idlist[i]][col].count()
                dataset.loc[dataset[col]==idlist[i], col+'_rank'] =\
                    dataset[
                        dataset[col]==idlist[i]].context_timestamp.rank()/countNum
                maxtime = dataset[
                        dataset[col]==idlist[i]].context_timestamp.max()
                mintime = dataset[
                        dataset[col]==idlist[i]].context_timestamp.min()
                dataset.loc[dataset[col]==idlist[i], col+'_timeperiod'] =\
                    (maxtime - mintime).seconds
    return dataset


#lastTime = -1
#def getLastTimeDiff(x):
#    global lastTime
#    if type(lastTime) == int:
#        lastTime = x
#        return -1
#    temp = lastTime
#    lastTime = x
#    return (x-temp).seconds
#
#
#def mergeIdTimeDiffInfo(dataset, collist):
#    global lastTime
#    dataset = dataset = dataset.sort_values(by='context_timestamp')
#    for col in collist:
#        print('processing:\t', col)
#        idlist = list(dataset[col].unique())
#        dataset[col+'_timeDiff'] = 0
#        for i in tqdm(range(len(idlist))):
#            lastTime = -1
#            dataset.loc[dataset[col]==idlist[i], col+'_timeDiff'] =\
#                    dataset[
#                        dataset[col]==idlist[i]].context_timestamp.apply(getLastTimeDiff)
#    return dataset

collist = ['item_brand_id', 'item_id', 'item_city_id', 'user_id',
           'user_occupation_id', 'user_gender_id', 'shop_id',
           'context_page_id', 'context_id', 'item_cat_id',
           ['user_id', 'item_id'], ['user_id', 'shop_id'],
           ['user_id', 'item_cat_id'], ['user_gender_id', 'item_cat_id'],
           ['user_occupation_id', 'item_cat_id'], ['shop_id', 'item_cat_id'],
           ]

colRanklist = ['user_id', 'item_brand_id', 'item_id', 'shop_id']


#colMeandict = {
#        'user_id': ['shop_score_description', 'shop_score_delivery',
#                    'shop_score_service', 'shop_review_positive_rate',
#                    'shop_star_level', 'shop_review_num_level',
#                    'item_sales_level', 'item_price_level',
#                    'item_collected_level', 'item_pv_level']
#        }

for day in range(2, 8):
    info_list = []
    for col in collist:
        if type(col) == str:
            info_list.append(getColInfo([col], allData[allData.day == day]))
        elif type(col) == list:
            info_list.append(getColInfo(col, allData[allData.day == day]))
            
    for col in collist:
        if type(col) == str:
            info_list.append(getColInfo([col], allData[allData.day == day-1],
                                        rate_col=True, nameAdd='-1_'))
        elif type(col) == list:
            info_list.append(getColInfo(col, allData[allData.day == day-1],
                                        rate_col=True, nameAdd='-1_'))
            
    for col in collist:
        if type(col) == str:
            info_list.append(getColInfo([col], allData[allData.day == day-2],
                                        rate_col=True, nameAdd='-2_'))
        elif type(col) == list:
            info_list.append(getColInfo(col, allData[allData.day == day-2],
                                        rate_col=True, nameAdd='-2_'))
#    for key in  colMeandict:
#          for col in colMeandict[key]:
#              print(col)
#              info_list.append(getColMeanInfo([key],
#                               allData[np.array(allData.day < day)&
#                                       np.array(allData.day >= day-2)&
#                                       np.array(allData.is_trade==1)],
#                               col))
#    for key in  colMeandict:
#          for col in colMeandict[key]:
#              print(col)
#              info_list.append(getColMeanInfo([key],
#                               pd.concat([
#                                sampleData[np.array(sampleData.day == day)],
#                                allData[np.array(allData.day < day)&
#                                        np.array(allData.day >= day-2)]
#                                ]),
#                               col))
          
    dataset = allData[allData.day == day]
    dataset = mergeInfo(dataset, info_list)
        
    dataset = mergeIdRankInfo(dataset, colRanklist)
    dataset.to_csv(
            './ProcessedData/day'+str(day)+'.csv', index=False)
    gc.collect()
    
    
    



# =============================================================================
# 这个模块的作用是把上面的那些day.csv转换成模型训练用的csv
# =============================================================================
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

valid = allData[allData.day == 6].drop(['day'], axis=1)


train = allData[allData.day <= 6]
train = train[train.day > 2].drop(['day'], axis=1)

test = allData[allData.day == 7][train.drop(['is_trade'], axis=1).columns]


pretrain.to_csv('./ProcessedData/pretrain.csv', index=False)
valid.to_csv('./ProcessedData/valid.csv', index=False)
train.to_csv('./ProcessedData/train.csv', index=False)
test.to_csv('./ProcessedData/test.csv', index=False)

gc.collect()